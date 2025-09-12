# cim_converter/cim_to_csv_converter.py
import logging
from pathlib import Path
import pandas as pd
import networkx as nx
from cimgraph.models import FeederModel
import cimgraph.utils as utils
from distopf.cim_importer.processors import (
    LineProcessor,
    SwitchProcessor,
    TransformerProcessor,
    RegulatorProcessor,
    CapacitorProcessor,
    GeneratorProcessor,
    BusProcessor,
)
from distopf.cim_importer.validators import TopologyValidator

_log = logging.getLogger(__name__)


def load_cim_model(
    cim_file: str | Path, s_base: float = 1e6
) -> dict[str, pd.DataFrame]:
    converter = CIMToCSVConverter(cim_file, s_base=s_base)
    return converter.convert(validate=True)


class CIMToCSVConverter:
    """Main converter class that orchestrates the conversion process."""

    def __init__(self, cim_file: str | Path, s_base: float = 1e6):
        self.cim_file = Path(cim_file)
        self.s_base = s_base
        self.network = None
        self.processors = {
            "line": LineProcessor(s_base),
            "switch": SwitchProcessor(s_base),
            "xfm": TransformerProcessor(s_base),
            "reg": RegulatorProcessor(s_base),
            "cap": CapacitorProcessor(s_base),
            "gen": GeneratorProcessor(s_base),
            "bus": BusProcessor(s_base),
        }
        self.validator = TopologyValidator()

    def load_network(self):
        """Load CIM network from file."""
        import os

        os.environ["CIMG_CIM_PROFILE"] = "cimhub_2023"
        from cimgraph.databases import XMLFile
        import cimgraph.data_profile.cimhub_2023 as cim

        file = XMLFile(filename=self.cim_file)
        self.network = FeederModel(container=cim.Feeder(), connection=file)
        utils.get_all_data(self.network)
        _log.info("Network loaded successfully")

    def convert(self, validate: bool = True) -> dict:
        """Main conversion method."""
        if self.network is None:
            self.load_network()
        assert self.network is not None
        _log.info("Starting conversion process...")

        # Process all components into dataframes
        data = dict(
            bus_data=pd.DataFrame(self.processors["bus"].process(self.network)),
            branch_data=self._process_branch_data(),
            gen_data=pd.DataFrame(self.processors["gen"].process(self.network)),
            cap_data=pd.DataFrame(self.processors["cap"].process(self.network)),
            reg_data=pd.DataFrame(self.processors["reg"].process(self.network)),
        )

        validation_result = None
        if validate:
            validation_result = self.validator.validate_tree_topology(
                data["branch_data"]
            )
            if validation_result and not validation_result["valid"]:
                _log.warning(
                    f"Topology validation failed: {validation_result['issues']}"
                )
            if validation_result and len(validation_result["warnings"]) > 0:
                _log.warning(f"Topology warnings: {validation_result['warnings']}")

        data = self._link_dataframes(data)
        if len(data["gen_data"]) > 0:
            data["gen_data"] = self._correct_generator_phases(
                data["bus_data"], data["gen_data"]
            )
            data["gen_data"] = self._aggregate_generators(data["gen_data"])
        data["bus_data"] = self._convert_secondary_loads(data["bus_data"])

        return data

    def _process_branch_data(self) -> pd.DataFrame:
        """Process all branch data (lines, switches, transformers)."""
        _log.info("Processing branch data...")
        all_data = []
        all_data.extend(self.processors["line"].process_branch(self.network))
        all_data.extend(self.processors["switch"].process_branch(self.network))
        all_data.extend(self.processors["xfm"].process_branch(self.network))
        all_data.extend(self.processors["reg"].process_branch(self.network))
        _log.info(f"Processed {len(all_data)} branch entries")
        return pd.DataFrame(all_data)

    def _fix_bus_phases_from_branches(
        self, bus_data: pd.DataFrame, branch_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Post-process bus phases based on connected branch phases using pandas operations."""
        if branch_data.empty:
            return bus_data

        # Create phase mapping from 'to' buses
        to_phases = branch_data[["tb", "phases"]].rename(columns={"tb": "id"})
        # For bus id 1 (swing bus), use 'from' connection phases
        fb_1 = branch_data[branch_data["fb"] == 1][["fb", "phases"]].rename(
            columns={"fb": "id"}
        )
        # Combine and merge with bus data
        phase_updates = pd.concat([fb_1, to_phases]).drop_duplicates("id", keep="last")
        bus_data = bus_data.merge(
            phase_updates, on="id", how="left", suffixes=("", "_new")
        )
        bus_data["phases"] = bus_data["phases_new"]
        bus_data.drop("phases_new", axis=1, inplace=True)
        return bus_data

    def _fix_downstream_phase_consistency(
        self, branch_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Fix cases where downstream branches have more phases than upstream branches."""
        if branch_data.empty:
            return branch_data

        # Create upstream mapping (fb -> upstream_phases)
        upstream_map = branch_data.set_index("tb")["phases"]
        # Map upstream phases and apply intersection
        branch_data["upstream_phases"] = branch_data["fb"].map(upstream_map)
        mask = branch_data["upstream_phases"].notna()
        branch_data.loc[mask, "phases"] = branch_data.loc[mask].apply(
            lambda x: "".join(
                sorted(set(x["phases"]).intersection(set(x["upstream_phases"])))
            ),
            axis=1,
        )
        branch_data.drop("upstream_phases", axis=1, inplace=True)
        return branch_data

    def _link_dataframes(self, data):
        bus_df = data["bus_data"]
        branch_df = data["branch_data"]
        gen_df = data["gen_data"]
        cap_df = data["cap_data"]
        reg_df = data["reg_data"]
        """Create DFS-based integer IDs and link all dataframes."""
        if bus_df.empty or branch_df.empty:
            _log.warning("Bus or branch data is empty, cannot perform linking.")
            return bus_df, branch_df, reg_df, cap_df, gen_df

        swing_bus_series = bus_df[bus_df["bus_type"] == "SWING"]
        if swing_bus_series.empty:
            raise ValueError(
                "No SWING bus found. Cannot determine network root for DFS."
            )

        root_bus_name = swing_bus_series["name"].iloc[0]
        graph = nx.Graph()
        for _, row in branch_df.iterrows():
            graph.add_edge(row["from_name"], row["to_name"])

        if root_bus_name not in graph:
            _log.warning(
                f"Swing bus '{root_bus_name}' not found in the graph constructed from branches. "
                "Using arbitrary node order."
            )
            all_nodes = list(bus_df["name"])
            dfs_nodes = [node for node in all_nodes if node in graph]
            if root_bus_name not in dfs_nodes:
                dfs_nodes.insert(0, root_bus_name)
        if root_bus_name in graph:
            dfs_nodes = list(nx.dfs_preorder_nodes(graph, source=root_bus_name))

        bus_id_map = {name: i + 1 for i, name in enumerate(dfs_nodes)}
        bus_df["id"] = bus_df["name"].map(bus_id_map)
        bus_df.dropna(subset=["id"], inplace=True)
        bus_df["id"] = bus_df["id"].astype(int)

        if not branch_df.empty:
            branch_df["fb"] = branch_df["from_name"].map(bus_id_map).astype(int)
            branch_df["tb"] = branch_df["to_name"].map(bus_id_map).astype(int)

        if not reg_df.empty:
            reg_df["fb"] = reg_df["from_name"].map(bus_id_map).astype(int)
            reg_df["tb"] = reg_df["to_name"].map(bus_id_map).astype(int)

        if not cap_df.empty:
            cap_df["id"] = cap_df["id"].map(bus_id_map).astype(int)

        if not gen_df.empty:
            gen_df["id"] = gen_df["id"].map(bus_id_map).astype(int)
        try:
            branch_df = self._fix_downstream_phase_consistency(branch_df)
        except pd.errors.InvalidIndexError:
            _log.warning("Cannot fix phase consistency due to network having loops.")
        bus_df = self._fix_bus_phases_from_branches(bus_df, branch_df)
        bus_df = bus_df.sort_values("id")
        data["bus_data"] = bus_df
        data["branch_data"] = branch_df
        data["gen_data"] = gen_df
        data["cap_data"] = cap_df
        data["reg_data"] = reg_df
        return data

    def _aggregate_generators(self, gen_df) -> pd.DataFrame:
        if len(gen_df) == 0:
            return gen_df
        """Aggregate generators by bus using pandas groupby."""
        original_count = len(gen_df)
        # Define aggregation functions for each column
        agg_dict = {
            "mrid": lambda x: "|".join(x),  # Combine mRIDs with pipe separator
            "name": lambda x: f"AggGen_{x.iloc[0]}_{len(x)}units"
            if len(x) > 1
            else x.iloc[0],
            "pa": "sum",
            "pb": "sum",
            "pc": "sum",
            "qa": "sum",
            "qb": "sum",
            "qc": "sum",
            "s_base": "sum",  # Sum the individual capacities
            "sa_max": "sum",
            "sb_max": "sum",
            "sc_max": "sum",
            "phases": lambda x: "".join(
                sorted(set("".join(x)))
            ),  # Combine unique phases
            "qa_max": "sum",
            "qb_max": "sum",
            "qc_max": "sum",
            "qa_min": "sum",
            "qb_min": "sum",
            "qc_min": "sum",  # Note: sums minimums (typically negative)
            "control_variable": "first",  # Use first control variable
        }
        # Only aggregate columns that exist in the dataframe
        available_agg_dict = {
            col: agg_func for col, agg_func in agg_dict.items() if col in gen_df.columns
        }
        # Group by bus ID and aggregate
        aggregated_gen_df = gen_df.groupby("id", as_index=False).agg(available_agg_dict)
        _log.info(
            f"Generator aggregation: {original_count} individual -> {len(aggregated_gen_df)} aggregated"
        )
        return aggregated_gen_df

    def _get_generator_columns(self):
        """Get standard generator data columns for ordering."""
        return [
            "mrid",
            "id",
            "name",
            "pa",
            "pb",
            "pc",
            "qa",
            "qb",
            "qc",
            "s_base",
            "sa_max",
            "sb_max",
            "sc_max",
            "phases",
            "qa_max",
            "qb_max",
            "qc_max",
            "qa_min",
            "qb_min",
            "qc_min",
            "ps1",
            "ps2",
            "qs1",
            "qs2",
            "ss1_max",
            "ss2_max",
            "qs1_max",
            "qs2_max",
            "qs1_min",
            "qs2_min",
            "control_variable",
        ]

    def _correct_generator_phases(self, bus_df: pd.DataFrame, gen_df: pd.DataFrame):
        gen_df = self._convert_secondary_gens(bus_df, gen_df)
        gen_df = self._correct_generators_without_phases(bus_df, gen_df)
        gen_df = self._distribute_phase_parameters(gen_df)
        return gen_df

    def _correct_generators_without_phases(
        self, bus_df: pd.DataFrame, gen_df: pd.DataFrame
    ):
        if len(gen_df) == 0:
            return gen_df
        mask = gen_df.phases.str.len() == 0
        gen_df.loc[mask, "phases"] = gen_df.loc[mask, "id"].apply(
            lambda x: bus_df.loc[bus_df.id == x, "phases"].to_list()[0]
        )
        num_phases = gen_df.loc[mask, "phases"].str.len()
        mask_a = mask & gen_df.phases.str.contains("a")
        mask_b = mask & gen_df.phases.str.contains("b")
        mask_c = mask & gen_df.phases.str.contains("c")
        gen_df.loc[mask_a, "pa"] = gen_df.loc[mask_a, "p"] / num_phases
        gen_df.loc[mask_b, "pb"] = gen_df.loc[mask_b, "p"] / num_phases
        gen_df.loc[mask_c, "pc"] = gen_df.loc[mask_c, "p"] / num_phases
        gen_df.loc[mask_a, "qa"] = gen_df.loc[mask_a, "q"] / num_phases
        gen_df.loc[mask_b, "qb"] = gen_df.loc[mask_b, "q"] / num_phases
        gen_df.loc[mask_c, "qc"] = gen_df.loc[mask_c, "q"] / num_phases
        return gen_df

    def _distribute_phase_parameters(self, gen_df: pd.DataFrame):
        if len(gen_df) == 0:
            return gen_df
        num_phases = gen_df.loc[:, "phases"].str.len()
        mask_a = gen_df.phases.str.contains("a")
        mask_b = gen_df.phases.str.contains("b")
        mask_c = gen_df.phases.str.contains("c")
        gen_df.loc[mask_a, "sa_max"] = gen_df.loc[mask_a, "s_max"] / num_phases
        gen_df.loc[mask_b, "sb_max"] = gen_df.loc[mask_b, "s_max"] / num_phases
        gen_df.loc[mask_c, "sc_max"] = gen_df.loc[mask_c, "s_max"] / num_phases
        gen_df.loc[mask_a, "qa_max"] = gen_df.loc[mask_a, "q_max"] / num_phases
        gen_df.loc[mask_b, "qb_max"] = gen_df.loc[mask_b, "q_max"] / num_phases
        gen_df.loc[mask_c, "qc_max"] = gen_df.loc[mask_c, "q_max"] / num_phases
        gen_df.loc[mask_a, "qa_min"] = gen_df.loc[mask_a, "q_min"] / num_phases
        gen_df.loc[mask_b, "qb_min"] = gen_df.loc[mask_b, "q_min"] / num_phases
        gen_df.loc[mask_c, "qc_min"] = gen_df.loc[mask_c, "q_min"] / num_phases
        return gen_df

    def _convert_secondary_gens(self, bus_df: pd.DataFrame, gen_df: pd.DataFrame):
        if len(gen_df) == 0:
            return gen_df
        if not (
            "ps1" in gen_df.keys()
            and "ps2" in gen_df.keys()
            and "qs1" in gen_df.keys()
            and "qs2" in gen_df.keys()
        ):
            return gen_df
        mask = (
            (gen_df.ps1.abs() > 0)
            | (gen_df.ps2.abs() > 0)
            | (gen_df.qs1.abs() > 0)
            | (gen_df.qs2.abs() > 0)
        )
        # change s1s2 to a, b, or c primary phase
        gen_df.loc[mask, "phases"] = gen_df.loc[mask, "id"].apply(
            lambda x: bus_df.loc[bus_df.id == x, "phases"].to_list()[0]
        )
        # move values from s1 s2 to primary phase columns
        mask_a = mask & gen_df.phases.str.contains("a")
        mask_b = mask & gen_df.phases.str.contains("b")
        mask_c = mask & gen_df.phases.str.contains("c")
        gen_df.loc[mask_a, "pa"] = gen_df.loc[mask_a, ["ps1", "ps2"]].sum(axis=1)
        gen_df.loc[mask_b, "pb"] = gen_df.loc[mask_b, ["ps1", "ps2"]].sum(axis=1)
        gen_df.loc[mask_c, "pc"] = gen_df.loc[mask_c, ["ps1", "ps2"]].sum(axis=1)
        gen_df.loc[mask_a, "qa"] = gen_df.loc[mask_a, ["qs1", "qs2"]].sum(axis=1)
        gen_df.loc[mask_b, "qb"] = gen_df.loc[mask_b, ["qs1", "qs2"]].sum(axis=1)
        gen_df.loc[mask_c, "qc"] = gen_df.loc[mask_c, ["qs1", "qs2"]].sum(axis=1)
        return gen_df

    def _convert_secondary_loads(self, bus_df: pd.DataFrame):
        """Transfer loads from s1, and s2 secondary phases to the appropriate phase."""
        mask = (
            (bus_df.pl_s1 > 0)
            | (bus_df.pl_s2 > 0)
            | (bus_df.ql_s1 > 0)
            | (bus_df.ql_s2 > 0)
        )
        _df = bus_df.loc[mask]
        for phase, _id, p1, p2, q1, q2 in zip(
            _df.phases, _df.id, _df.pl_s1, _df.pl_s2, _df.ql_s1, _df.ql_s2
        ):
            bus_df.loc[bus_df.id == int(_id), [f"pl_{phase}"]] = p1 + p2
            bus_df.loc[bus_df.id == int(_id), [f"ql_{phase}"]] = q1 + q2
        return bus_df

    def save(self, results: dict[str, pd.DataFrame], output_dir: str | Path):
        """Save processed data to CSV files."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        bus_cols = self._get_bus_columns()
        branch_cols = self._get_branch_columns()

        bus_data = results["bus_data"]
        # Save bus data
        if bus_data.empty:
            pd.DataFrame(columns=bus_cols).to_csv(
                self.output_dir / "bus_data.csv", index=False
            )
        if not bus_data.empty:
            bus_data.to_csv(
                self.output_dir / "bus_data.csv", index=False, columns=bus_cols
            )
        _log.info(f"Saved bus data to {self.output_dir / 'bus_data.csv'}")

        branch_data = results["branch_data"]
        # Save branch data
        if branch_data.empty:
            pd.DataFrame(columns=branch_cols).to_csv(
                self.output_dir / "branch_data.csv", index=False
            )
        if not branch_data.empty:
            branch_data.to_csv(
                self.output_dir / "branch_data.csv", index=False, columns=branch_cols
            )
        _log.info(f"Saved branch data to {self.output_dir / 'branch_data.csv'}")

        regulator_data = results["reg_data"]
        if not regulator_data.empty:
            regulator_data.to_csv(self.output_dir / "reg_data.csv", index=False)
        capacitor_data = results["cap_data"]
        if not capacitor_data.empty:
            capacitor_data.to_csv(self.output_dir / "cap_data.csv", index=False)
        generator_data = results["gen_data"]
        if not generator_data.empty:
            generator_data.to_csv(self.output_dir / "gen_data.csv", index=False)
            _log.info(f"Saved generator data to {self.output_dir / 'gen_data.csv'}")

    def _get_branch_columns(self):
        """Get standard branch data columns for ordering."""
        return [
            "name",
            "fb",
            "tb",
            "from_name",
            "to_name",
            "raa",
            "rab",
            "rac",
            "rbb",
            "rbc",
            "rcc",
            "xaa",
            "xab",
            "xac",
            "xbb",
            "xbc",
            "xcc",
            "type",
            "status",
            "s_base",
            "v_ln_base",
            "z_base",
            "phases",
            "length",
        ]

    def _get_bus_columns(self):
        """Get standard bus data columns for ordering."""
        return [
            "mrid",
            "id",
            "name",
            "pl_a",
            "ql_a",
            "pl_b",
            "ql_b",
            "pl_c",
            "ql_c",
            "bus_type",
            "v_a",
            "v_b",
            "v_c",
            "v_ln_base",
            "s_base",
            "v_min",
            "v_max",
            "cvr_p",
            "cvr_q",
            "phases",
            "latitude",
            "longitude",
        ]

from distopf.cim_importer.processors.base_processor import BaseProcessor
import cimgraph.data_profile.cimhub_2023 as cim
from distopf.cim_importer.utils import PhaseUtils


class GeneratorProcessor(BaseProcessor):
    """Processor for generator objects (PowerElectronicsConnection, EnergySource, etc.)."""

    def process(self, network) -> list[dict]:
        """Process all generator types."""
        results = []
        # Process PowerElectronicsConnection (PV, Battery, etc.)
        results.extend(self._process_power_electronics_connections(network))
        # Process EnergySource (may be generators or sources)
        results.extend(self._process_energy_sources(network))
        return results

    def _process_power_electronics_connections(self, network) -> list[dict]:
        """Process PowerElectronicsConnection objects."""
        results = []
        for pec in network.graph.get(cim.PowerElectronicsConnection, {}).values():
            gen_data = self._process_power_electronics_connection(pec)
            if gen_data:
                results.append(gen_data)
        return results

    def _process_power_electronics_connection(self, pec) -> dict | None:
        """Process individual PowerElectronicsConnection."""
        if not pec.Terminals or len(pec.Terminals) == 0:
            return None

        s_base_gen = (
            float(pec.ratedS) if hasattr(pec, "ratedS") and pec.ratedS else self.s_base
        )

        bus_name = pec.Terminals[0].ConnectivityNode.name
        gen_name = pec.name

        # Convert power and limits to system per-unit
        total_p_pu = float(pec.p) / self.s_base if hasattr(pec, "p") and pec.p else 0.0
        total_q_pu = float(pec.q) / self.s_base if hasattr(pec, "q") and pec.q else 0.0
        max_q_pu = (
            (float(pec.maxQ) / self.s_base)
            if hasattr(pec, "maxQ") and pec.maxQ
            else 0.0
        )
        min_q_pu = (
            (float(pec.minQ) / self.s_base)
            if hasattr(pec, "minQ") and pec.minQ
            else 0.0
        )
        rated_s_pu = (
            (float(pec.ratedS) / self.s_base)
            if hasattr(pec, "ratedS") and pec.ratedS
            else 0.0
        )

        gen_data = {
            "mrid": pec.mRID,
            "id": bus_name,  # Use bus name (will be mapped to integer later)
            "name": gen_name,
            "pa": 0.0,
            "pb": 0.0,
            "pc": 0.0,
            "qa": 0.0,
            "qb": 0.0,
            "qc": 0.0,
            "s_base": s_base_gen,
            "sa_max": 0.0,
            "sb_max": 0.0,
            "sc_max": 0.0,
            "phases": "",
            "qa_max": 0.0,
            "qb_max": 0.0,
            "qc_max": 0.0,
            "qa_min": 0.0,
            "qb_min": 0.0,
            "qc_min": 0.0,
            "p": total_p_pu,
            "q": total_q_pu,
            "s_max": rated_s_pu,
            "s_min": -rated_s_pu,
            "q_max": max_q_pu,
            "q_min": min_q_pu,
            "ps1": 0.0,
            "ps2": 0.0,
            "qs1": 0.0,
            "qs2": 0.0,
            "ss1_max": 0.0,
            "ss2_max": 0.0,
            "qs1_max": 0.0,
            "qs2_max": 0.0,
            "qs1_min": 0.0,
            "qs2_min": 0.0,
            "control_variable": "PQ",
        }

        # Process phase-specific data
        phase_data = {}
        active_phases_for_limits = set()
        if (
            hasattr(pec, "PowerElectronicsConnectionPhases")
            and pec.PowerElectronicsConnectionPhases
        ):
            phase_conn: cim.PowerElectronicsConnectionPhase
            for phase_conn in pec.PowerElectronicsConnectionPhases:
                phase_letter = PhaseUtils.get_phase_str(phase_conn.phase)
                if not phase_letter:
                    continue
                phase_p = (
                    float(phase_conn.p) / self.s_base
                    if hasattr(phase_conn, "p") and phase_conn.p
                    else 0.0
                )
                phase_q = (
                    float(phase_conn.q) / self.s_base
                    if hasattr(phase_conn, "q") and phase_conn.q
                    else 0.0
                )
                phase_data[phase_letter] = {"p": phase_p, "q": phase_q}

                active_phases_for_limits.add(phase_letter)

        # Distribute power across phases
        if phase_data:
            abc_phases, s_phases = set(), set()
            for phase_letter, data in phase_data.items():
                if phase_letter in ["s1", "s2"]:
                    s_phases.add(phase_letter)
                if phase_letter in "abc":
                    abc_phases.add(phase_letter)
                gen_data[f"p{phase_letter}"], gen_data[f"q{phase_letter}"] = (
                    data["p"],
                    data["q"],
                )
            gen_data["phases"] = (
                f"{''.join(sorted(abc_phases))}{''.join(sorted(s_phases))}"
            )
        # else:
        #     gen_data["pa"] = gen_data["pb"] = gen_data["pc"] = total_p_pu / 3.0
        #     gen_data["qa"] = gen_data["qb"] = gen_data["qc"] = total_q_pu / 3.0
        #     gen_data["phases"] = "abc"
        #     if total_p_pu != 0 or total_q_pu != 0:
        #         active_phases_for_limits = {"a", "b", "c"}

        # Distribute limits (in system p.u.) across active phases
        num_phases = len(active_phases_for_limits)
        if rated_s_pu > 0 and num_phases > 0:
            s_max_pu_per_phase = rated_s_pu / num_phases
            q_max_pu_per_phase = max_q_pu / num_phases
            q_min_pu_per_phase = min_q_pu / num_phases
            for phase_char in active_phases_for_limits:
                gen_data[f"s{phase_char}_max"] = s_max_pu_per_phase
                gen_data[f"q{phase_char}_max"] = q_max_pu_per_phase
                gen_data[f"q{phase_char}_min"] = q_min_pu_per_phase

        return gen_data

    def _process_energy_sources(self, network) -> list[dict]:
        results = []
        for source in network.graph.get(cim.EnergySource, {}).values():
            if hasattr(source, "Terminals") and source.Terminals:
                if "sourcebus" in source.Terminals[0].ConnectivityNode.name.lower():
                    continue
            gen_data = self._process_energy_source(source)
            if gen_data:
                results.append(gen_data)
        return results

    def _process_energy_source(self, source) -> dict | None:
        if not source.Terminals or len(source.Terminals) == 0:
            return None
        bus_name = source.Terminals[0].ConnectivityNode.name

        gen_data = {
            "mrid": source.mRID,
            "id": bus_name,  # Use bus name (will be mapped to integer later)
            "name": f"Generator_{source.mRID[:8]}",
            "pa": 0.0,
            "pb": 0.0,
            "pc": 0.0,
            "qa": 0.0,
            "qb": 0.0,
            "qc": 0.0,
            "s_base": self.s_base,
            "sa_max": 0.0,
            "sb_max": 0.0,
            "sc_max": 0.0,
            "phases": "abc",
            "qa_max": 0.0,
            "qb_max": 0.0,
            "qc_max": 0.0,
            "qa_min": 0.0,
            "qb_min": 0.0,
            "qc_min": 0.0,
            "ps1": 0.0,
            "ps2": 0.0,
            "qs1": 0.0,
            "qs2": 0.0,
            "ss1_max": 0.0,
            "ss2_max": 0.0,
            "qs1_max": 0.0,
            "qs2_max": 0.0,
            "qs1_min": 0.0,
            "qs2_min": 0.0,
            "control_variable": "PQ",
        }
        if hasattr(source, "activePower") and source.activePower:
            total_p = float(source.activePower) / self.s_base
            gen_data["pa"] = gen_data["pb"] = gen_data["pc"] = total_p / 3.0
        if hasattr(source, "reactivePower") and source.reactivePower:
            total_q = float(source.reactivePower) / self.s_base
            gen_data["qa"] = gen_data["qb"] = gen_data["qc"] = total_q / 3.0
        return gen_data

    def _get_phase_str(self, phase_code) -> str | None:
        phase_str = str(phase_code.value).lower()
        if "s1" in phase_str:
            return "s1"
        elif "s2" in phase_str:
            return "s2"
        elif "a" in phase_str:
            return "a"
        elif "b" in phase_str:
            return "b"
        elif "c" in phase_str:
            return "c"
        return None

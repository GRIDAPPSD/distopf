from distopf.pyomo_models.protocol import LindistModelProtocol
import pyomo.environ as pyo


def loss_objective_rule(model: LindistModelProtocol):
    """
    Calculate total system losses using the resistance parameters.
    For each branch-phase combination, calculates (P² + Q²) * R
    """
    total_loss = 0
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            total_loss += (model.p_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
            total_loss += (model.q_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
    return total_loss


loss_objective = pyo.Objective(
    rule=loss_objective_rule,
    sense=pyo.minimize,
)

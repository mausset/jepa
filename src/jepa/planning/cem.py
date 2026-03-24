from jepa.planning.base_planner import BasePlanner


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        wm,
        action_dim,
        pre_processor,
        horizon,
        population=1024,
        elite_frac=0.1,
        iterations=6,
        alpha=0.1,
        progress_bar=True,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.population = population
        self.elite = max(1, int(population * elite_frac))
        self.iterations = iterations
        self.alpha = alpha
        self.progress_bar = progress_bar
        self.device = next(wm.parameters()).device

    def plan(self, x):
        pass

import glob

class Setup(object):
    """Class responsible for startup and teardown ops."""
    def startup(self):
        """Perform all setup ops."""
        env_models = {}
        for expert in glob.glob("experts/*"):
            model = expert.strip().split("/")[1]
            env = model.split(".")[0]
            env_models[env] = model
        return env_models

    def teardown(self):
        """Perform all tear down ops."""
        pass


def behavioral_cloning(env_name, expert_policy):
    print(f"env_name: {env_name} -- expert_policy: {expert_policy}")
    pass

def dagger(env_name, expert_policy):
    print(f"env_name: {env_name} -- expert_policy: {expert_policy}")
    pass

if __name__ == '__main__':
    setup = Setup()
    env_models = setup.startup()

    # behavioral cloning runs
    for env, model in env_models.items():
        ret = behavioral_cloning(env_name=env, expert_policy=model)

    # dagger runs
    for env, model in env_models.items():
        ret = dagger(env_name=env, expert_policy=model)

    setup.teardown()
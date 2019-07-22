# Developed by Lorenzo Mambretti, Justin Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://github.com/jtwwang/hanabi/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied


def load_agent(agent_class):
    """
    import the module required for a specific class and returns
    the agent correspondent to that class
    """

    if agent_class == 'NewestCardAgent':
        from agents.newest_card_agent import NewestCardAgent
        agent = NewestCardAgent
    elif agent_class == 'NeuroEvoAgent':
        from agents.neuroEvo_agent import NeuroEvoAgent
        agent = NeuroEvoAgent
    elif agent_class == 'NNAgent':
        from agents.nn_agent import NNAgent
        agent = NNAgent
    elif agent_class == 'MCAgent':
        from agents.mc_agent import MCAgent
        agent = MCAgent
    elif agent_class == 'ProbabilisticAgent':
        from agents.probabilistic_agent import ProbabilisticAgent
        agent = ProbabilisticAgent
    elif agent_class == 'RainbowAgent':
        from agents.rainbow_agent_rl import RainbowAgent
        agent = RainbowAgent
    elif agent_class == 'RandomAgent':
        from agents.random_agent import RandomAgent
        agent = RandomAgent
    elif agent_class == 'SecondAgent':
        from agents.second_agent import SecondAgent
        agent = SecondAgent
    elif agent_class == 'SimpleAgent':
        from agents.simple_agent import SimpleAgent
        agent = SimpleAgent
    else:
        raise ValueError("Invalid agent_class %s" %agent_class)

    return agent

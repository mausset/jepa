# JEPA World Models

## What is a world model?

According to the bitter lesson the methods that scale the best are learning and search, however, note that learning is a form of amortized search (Maus 2025, Jonason 2026). To successfully train an agent to act in an environment we require two things, reward and experience. This is where world models come in. If value models are amortized reward, then world models are amortized experience.

A world model is mental model of how the world progresses. Assuming a value model, the world model helps the agent plan how to enter states of high value prior to taking action.

You can also use a world model with an end state in mind. We will consider this latter case and leave the usage of value models for another time.

## Latent world models

When planning with an end state in mind, we must be able to measure how close we are to said end state. In pixel space distances are meaningless, and so we must turn to a space where measurements are meaningful. A learned latent space is one such space. Indeed, the goal of latent world models is learn the world model in a latent space.

Let's assume that our model is a generator for the next state given current state that we can sample from. Planning from an initial state to an end state is done by searching over this path of conditional distributions such that our final sampled state is as close as possible to our target end state. Once again, we wrap back around to search; a smart search algorithm would make use of the meaningful distance defined by our latent space to prioritize paths that are approaching our target state.

When our search is complete we will have a path through state space. If we want to enact this plan we will need to translate it to actions, for which we require supervision. The neat part is that actions are the only part of the setup that requires supervision. The world model in itself only requires experience.

## Open questions

- Does it matter if we learn the latent space jointly or not?
- What type of generative model should we use?
- How do we plan through search? What search algorithms should we use?

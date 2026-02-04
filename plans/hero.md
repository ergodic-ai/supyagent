great. one more thing, when building the init, create a default "hero.yaml" agent.. this agent should be a planner agent, tell it to to:

1) assess the complexity of teh task
2) if the task is simple, return it right away
3) if it requires a very big implementation, the agent should write a plans/{plan_name}.md before implementing it
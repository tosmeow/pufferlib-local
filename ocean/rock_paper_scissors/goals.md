## Steps completed

- Architecture design was a 4x smaller than usual minGRU because of simplicity of the policies we knew towards which we wanted to converge.

- Trained rock paper scissors environment in self-play, which eventually converged to Nash equilibrium: indifferent of recent actions, always plays uniform at random across the 3 different actions.

## Next steps to explore

- No longer self-playing: want to hard-code various agent policies: players with different than the Nash equilibrium behavior but indifferent to our behavior, players that draw their actions based on what we have last done.

- Study then the equilibrium compared to what I'd expect on paper.
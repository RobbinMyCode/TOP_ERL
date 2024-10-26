# Transformer-based Off-policy Episodic RL (TOP-ERL)
### Under review in ICLR25

<p align="center">
  <img src='./web_assets/Metaworld.gif' width="200" />
  <img src='./web_assets/Box_Pushing.gif' width="200" />
  <img src='./web_assets/rollout.png' width="243" />
</p>

<br><br>

<p align="center">
  <img src='./web_assets/results.png' width="643" />
</p>

<br><br>

## Episodic RL, What and Why?
Episodic Reinforcement Learning (ERL) [1, 4, 5] is a distinct RL family that emphasizes the maximization of returns over entire episodes, typically lasting several seconds, rather than optimizing the intermediate states during environment interactions. Unlike Step-based RL (SRL) [2, 3], ERL shifts the solution search from per-step actions to a parameterized trajectory space, leveraging techniques like Movement Primitives (MPs) [6, 7, 8] for generating action sequences. This approach enables a broader exploration horizon [4], captures trajectory statistics [9], and ensures smooth transitions between re-planning phases [10].

<p align="center">Exploration Strategies Comparison, SRL vs. ERL [9]</p>

<table align="center">
  <tr>
    <td align="center">
      <img src='./web_assets/SRL.png' width="300" /><br>
      <em>Step-based RL explores per action step</em>
    </td>
    <td align="center">
      <img src='./web_assets/ERL.png' width="300" /><br>
      <em>Episodic RL has consistent exploration</em>
    </td>
  </tr>
</table>



## Use Movement Primitives for Trajectory Generation
Episodic RL often uses the movement primitves (MPs) as a paramterized trajectory generator. In TOP-ERL, we use the ProDMP [8] for fast computation and better initial condition enforcement. A simple illustration of using MPs can be seen as follows:

<p align="center">
  <img src='./web_assets/mp_demo.gif' width="600" /><br>
  <em>MP generates a trajectory (upper curve) by manipulating the basis functions (lower curves)</em>
</p>

## Use Transformer as an Action Sequence Critic
In the literature, most of the combinations of RL and Transformer focus on off-policy, model-based and POMDP settings. Directly using tranformer in online RL for acition sequence value prediction remains highly unexplored. In TOP-ERL, we utilize Transformers as an action sequence value predictor, training it via the N-step future returns. To ensure stable critic learning, we adapt the trajectory segmentation strategy in [9] by splitting the long trajectory into sub-sequences of varying lengths.

<p align="center">
  <img src='./web_assets/critic_animation_gif.gif' width="900" /><br>
  <em>TOP-ERL utilizes a transformer critic that predicts the value of executing a sub-sequence of actions from the beginning of the segment state. </em>
</p>


<br><br>
### References
[1] Darrell Whitley, Stephen Dominic, Rajarshi Das, and Charles W Anderson. Genetic reinforcement learning for neurocontrol problems. Machine Learning, 13:259–284, 1993.

[2] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[3] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. PMLR, 2018a.

[4] Jens Kober and Jan Peters. Policy search for motor primitives in robotics. NIPS, 2008.

[5] Jan Peters and Stefan Schaal. Reinforcement learning of motor skills with policy gradients. Neural networks, 21(4):682–697, 2008.

[6] Stefan Schaal. Dynamic movement primitives-a framework for motor control in humans and humanoid robotics. In Adaptive motion of animals and machines, pp. 261–280. Springer, 2006.

[7] Alexandros Paraschos, Christian Daniel, Jan Peters, and Gerhard Neumann. Probabilistic movement primitives. Advances in neural information processing systems, 26, 2013.

[8] Ge Li, Zeqi Jin, Michael Volpp, Fabian Otto, Rudolf Lioutikov, and Gerhard Neumann. Prodmp:A unified perspective on dynamic and probabilistic movement primitives. IEEE RA-L, 2023.

[9] Ge Li, Hongyi Zhou, Dominik Roth, Serge Thilges, Fabian Otto, Rudolf Lioutikov, and Gerhard Neumann. Open the black box: Step-based policy updates for temporally-correlated episodic reinforcement learning. ICLR 2024.

[10] Fabian Otto, Hongyi Zhou, Onur Celik, Ge Li, Rudolf Lioutikov, and Gerhard Neumann. Mp3: Movement primitive-based (re-) planning policy. arXiv preprint arXiv:2306.12729, 2023.


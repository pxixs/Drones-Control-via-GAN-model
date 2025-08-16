# Drone Swarm Navigation via Mean-Field Games (GAN/ResNet)

The controller learns a value function and an optimal feedback policy that:

1. Avoids obstacles
2. Avoids inter-drone collisions
3. Maintains a target formation
4. Reaches a terminal goal

---

## Highlights
- **Mean-Field Control / MFG:** model the swarm as a density $\rho(x,t)$ in $\mathbb{R}^3$ governed by a Fokker–Planck equation.
- **Physics-guided learning:** adversarial (GAN-style) training with a **generator** for trajectories and a **critic** $\phi$ (value function) that enforces FP/HJB residuals.
- **Feedback controller:** $u^\* = -\nabla \phi(x,t)$ derived from the Hamilton–Jacobi–Bellman optimality condition.
- **Task costs:**
  - **Obstacles** ($F_1$): strong penalty near obstacle boundaries.
  - **Inter-drone safety** ($F_2$): repulsion when distances $< \varepsilon$.
  - **Formation** ($F_3$): $L^1(\hat\rho_c, \rho_{0})$ between centered empirical density and target formation.
  - **Terminal goal** ($G$): mean position close to the target $x_f$.
- **Implementation details:**
  - PyTorch + light **ResNet** blocks.
  - Density via **KDE**; integrals on a **3D Riemann grid** (autograd-friendly).
  - YAML configs; `train.py` / `simulate.py` CLIs.
---

## Repository Structure
```text
.
├─ src
│  ├─ models
│  │  ├─ resblock.py         # Residual blocks / ResNet building blocks
│  │  ├─ generator.py        # Gθ: (z,t) → positions/trajectories
│  │  └─ discriminator.py    # φω: value function approximator
│  ├─ costs
│  │  ├─ obstacles.py        # Obstacle penalty F1
│  │  ├─ collisions.py       # Inter-drone penalty F2
│  │  └─ formation.py        # Formation loss F3 (KDE + L¹ on grid)
│  ├─ train.py               # Adversarial training loop
│  └─ simulate.py            # Rollouts & visualization
├─ main.py               # Optional entrypoint (calls train/simulate)
├─ Theoritical_Document(French)
└─ README.md
```
## Participants

Project carried out by **Mohssine Bakraoui**, **Andréa Bourelly**, **Aurélien Castillon**, **Joseph Combourieu**, **Imad El-Hassouni**, and **Mohamed-Réda Salhi**, within the MAP14 program at **École Polytechnique**, in collaboration with **MBDA**.

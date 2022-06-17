# MazeReinforcementLearningBasedOnQlearning
    Walk a simple maze based on qlearning(with walls, traps, starting points and ending points)
--------------------------------
## Dependent libraries
    Pytorch、Pandas、Numpy、Tkinter. Make sure you have these libraries installed 
--------------------------------
## How to run this demo_code using BasicQLearning method
```Python
"""
    Comment code A and B, which represents Reinforcement learning using method called BasicQLearning. Then run this code in main.py
"""
if __name__ == "__main__":
    maze = Maze(4, 4, 0, 2)
    e_greedy_increment = 0.01
    RL = QLearningTable(maze.get_actions(),e_greedy=0.5,e_greedy_increment=e_greedy_increment) # A
    mz = maze.get_maze()

    view = MazeView(mz, 0.1, 200 + 20*((0.9/e_greedy_increment)-1))
    view.after(100, play) # B
    view.mainloop()
```
## How to run this demo_code using DQN method
```Python
"""
    Comment code A and B, which represents Reinforcement learning using method DQN. Then run this code in main.py
"""
if __name__ == "__main__":
    maze = Maze(4, 4, 0, 2)
    e_greedy_increment = 0.01
    RL = DeepQNetwork(actions=maze.get_actions(), input=2, hidden=20, batch_size=150,
                      e_greedy_increment=e_greedy_increment) # A
    mz = maze.get_maze()

    view = MazeView(mz, 0.1, 200 + 20*((0.9/e_greedy_increment)-1))
    view.after(100, play2) # B
    view.mainloop()
```
--------------------------------
# REFERENCE
[1][莫凡强化学习教程BiliBili](https://www.bilibili.com/video/BV13W411Y75P?p=1&vd_source=7478d53454dfed00538f20c2bfa7123f)  
[2][ClownW](https://github.com/ClownW/Reinforcement-learning-with-PyTorch/tree/master/content/5_Deep_Q_Network)
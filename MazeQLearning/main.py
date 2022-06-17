from DQN import DeepQNetwork
from MazeSimulator import Maze
from BasicQLearningAlgorithm import QLearningTable
from MazeView import MazeView


def play():
    cnt = 0
    while True:
        maze.reset()
        view.reset()

        state = maze.get_state()
        while True:
            action = RL.choose_action(str(state))
            next_state, reward, done = maze.step(action)

            RL.learning(str(state), action, reward, str(next_state), done)

            state = next_state
            view.render(state[0], state[1])

            if done:
                break
        cnt += 1
        if cnt >= 200 and cnt % 20 == 0:
            RL.change_epsilon()


def play2():
    cnt = 0
    translate = {
        0: "up",
        1: "down",
        2: "left",
        3: "right"
    }
    while True:
        maze.reset()
        view.reset()

        state = maze.get_state()

        while True:
            action = RL.choose_action([i * 5 for i in state])
            # action = RL.choose_action(state)
            action_str = translate[action]

            next_state, reward, done = maze.step(action_str)

            RL.store_transition([i * 5 for i in state], action, reward, [i * 5 for i in next_state])
            # RL.store_transition(state, action, reward, next_state)

            if cnt >= 200 and cnt % 20 == 0:
                RL.learn()

            state = next_state
            view.render(state[0], state[1])

            if done:
                break
            cnt += 1


if __name__ == "__main__":
    maze = Maze(4, 4, 0, 2)
    e_greedy_increment = 0.01
    # RL = QLearningTable(maze.get_actions(),e_greedy=0.5,e_greedy_increment=e_greedy_increment)
    RL = DeepQNetwork(actions=maze.get_actions(), input=2, hidden=20, batch_size=150,
                      e_greedy_increment=e_greedy_increment)
    mz = maze.get_maze()

    view = MazeView(mz, 0.1, 200 + 20*((0.9/e_greedy_increment)-1))
    # view.after(100, play)
    view.after(100, play2)
    view.mainloop()

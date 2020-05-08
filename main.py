from ReplayMemory import *
from DQN import *
from AtariWrapper import *
from BreakAI import *
import tensorflow as tf
import tensorboard 
import time

def main():
    #Config hyperparam:
    MEM_SIZE = 1e6
    USE_PER = False
    BATCH_SIZE = 32
    PRIORITY_SCALE = 0.7     
    TOTAL_FRAMES = 30000000           # Total number of frames to train for
    MAX_EPISODE_LENGTH = 18000        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
    FRAMES_BETWEEN_EVAL = 100000      # Number of frames between evaluations
    EVAL_LENGTH = 10000               # Number of frames to evaluate for
    UPDATE_FREQ = 10000               # Number of actions chosen between updating the target network

    DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
    MIN_REPLAY_MEMORY_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent
    MEM_SIZE = 1000000                # The maximum size of the replay buffer

    MAX_NOOP_STEPS = 20               # Randomly perform this number of actions before every evaluation to give it an element of randomness
    UPDATE_FREQ = 4                   # Number of actions between gradient descent steps

    
    LOAD_FROM = None
    LOAD_REPLAY_MEMORY =True
    SAVE_PATH = 'Memory'
    WRITE_TENSORBOARD = True
    TENSORBOARD_DIR = 'tensorboard/'



    game_wrapper = AtariWrapper('Breakout-v0')
    print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # Build main and target networks
    MAIN_DQN = buildDQN(game_wrapper.env.action_space.n)
    TARGET_DQN = buildDQN(game_wrapper.env.action_space.n)

    ReplayMem = ReplayMemory(size=MEM_SIZE, use_per=USE_PER)
    agent = BreakAI(MAIN_DQN, TARGET_DQN, ReplayMem, game_wrapper.env.action_space.n, batch_size=BATCH_SIZE, use_per=USE_PER)

    # Training and evaluation
    if LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_MEMORY)

        # Apply information loaded from meta
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']

        print('Loaded')
    
    #######TRAINING########


    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                # Training

                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    game_wrapper.reset()
                    life_lost = True
                    episode_reward_sum = 0
                    for _ in range(MAX_EPISODE_LENGTH):
                        # Get action
                        action = agent.get_action(frame_number, game_wrapper.state)

                        # Take step
                        processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        # Add experience to replay memory
                        agent.add_experience(action=action,frame=processed_frame[:, :, 0],
                                            reward=reward,terminal=life_lost)

                        # Update agent bytt ut replay_buffer
                        if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_MEMORY_SIZE:
                            loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number, priority_scale=PRIORITY_SCALE)
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_MEMORY_SIZE:
                            agent.update_target_network()

                        # Break the loop when the game is over
                        if terminal:
                            terminal = False
                            break

                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0

                for _ in range(EVAL_LENGTH):
                    if terminal:
                        game_wrapper.reset(evaluation=True)
                        life_lost = True
                        episode_reward_sum = 0
                        terminal = False

                    # Breakout requires a "fire" action (action #1) to start the
                    # game each time a life is lost.
                    # Otherwise, the agent would sit around doing nothing.
                    action = 1 if life_lost else agent.get_action(frame_number, game_wrapper.state, evaluation=True)

                    # Step action
                    _, reward, terminal, life_lost = game_wrapper.step(action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    # On game-over
                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum
                # Print score and write to tensorboard
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # Save model
                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
            print('Saved.')

        

if __name__=="__main__":
    main()
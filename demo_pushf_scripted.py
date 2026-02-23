"""
Scripted demonstration collector for the Push-F task.
Generates demonstrations using a geometric waypoint planner (no GUI needed).
Saves data in Zarr format compatible with the training pipeline.
Also saves verification videos.

Usage:
    python demo_pushf_scripted.py -o data/pushf/pushf_demo.zarr -n 100
    python demo_pushf_scripted.py -o data/pushf/pushf_demo.zarr -n 5 --save_video
"""

import os
import numpy as np
import click
import cv2
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


def angle_diff(a, b):
    """Signed angle difference, result in [-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def move_toward(current, goal, max_step):
    """Return a point at most max_step away from current, toward goal."""
    diff = goal - current
    dist = np.linalg.norm(diff)
    if dist <= max_step or dist < 1e-6:
        return goal.copy()
    return current + diff / dist * max_step


def geometric_planner(block_pos, block_angle, goal_pos, goal_angle, agent_pos, rng):
    """
    Simple geometric planner for pushing F-block to goal pose.

    CRITICAL: Always move incrementally from current agent position.
    The PD controller is very stiff (k_p=100). Large target offsets
    cause violent accelerations.
    """
    pos_err = np.linalg.norm(goal_pos - block_pos)
    ang_err = abs(angle_diff(goal_angle, block_angle))

    noise = rng.uniform(-1.0, 1.0, size=2)

    # Compute push direction from block to goal
    if pos_err > 1e-6:
        push_dir = (goal_pos - block_pos) / pos_err
    else:
        push_dir = np.array([1.0, 0.0])

    # Check if agent is roughly behind the block (on far side from goal)
    agent_to_block = block_pos - agent_pos
    dist_ab = np.linalg.norm(agent_to_block)

    if pos_err > 8:
        # Phase 1: Move block toward goal
        # Target approach: move to far side of block, then push through

        # Where to approach from: far side of block, ~80px from center
        approach = block_pos - push_dir * 80

        # Check alignment: is agent behind the block?
        if dist_ab > 1e-6:
            alignment = np.dot(agent_to_block / dist_ab, push_dir)
        else:
            alignment = 1.0

        if alignment < 0.5 or dist_ab > 100:
            # Move to approach position (behind block)
            target = move_toward(agent_pos, approach, 5) + noise
        else:
            # Aligned behind block — push toward goal
            target = move_toward(agent_pos, goal_pos, 5) + noise
    else:
        # Phase 2: Rotate block
        if ang_err < 0.2:
            target = agent_pos + noise * 0.3
        else:
            rot_dir = np.sign(angle_diff(goal_angle, block_angle))

            # Approach perpendicular to block, then push tangentially
            perp_angle = block_angle + np.pi / 2
            perp = np.array([np.cos(perp_angle), np.sin(perp_angle)])
            tangent = np.array([-perp[1], perp[0]]) * rot_dir

            contact = block_pos + perp * 80

            if np.linalg.norm(agent_pos - contact) > 30:
                target = move_toward(agent_pos, contact, 5) + noise
            else:
                target = move_toward(agent_pos, agent_pos + tangent * 20, 5) + noise

    target = np.clip(target, 25, 487)
    return target


@click.command()
@click.option('-o', '--output', required=True, help='Output zarr path')
@click.option('-n', '--n_episodes', default=100, type=int, help='Number of episodes')
@click.option('--save_video', is_flag=True, help='Save verification videos')
@click.option('--video_dir', default='data/pushf/videos', help='Video output directory')
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('--max_steps', default=300, type=int, help='Max steps per episode')
def main(output, n_episodes, save_video, video_dir, render_size, max_steps):
    # Create replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # Create PushF env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params(block_shape='f')
    env = PushTKeypointsEnv(
        render_size=render_size,
        render_action=False,
        block_shape='f',
        **kp_kwargs
    )

    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    successes = 0
    total_rewards = []

    for ep_idx in range(n_episodes):
        seed = replay_buffer.n_episodes
        rng = np.random.default_rng(seed=seed + 1000)

        env.seed(seed)
        obs = env.reset()
        info = env._get_info()

        episode = []
        frames = []
        done = False

        for step in range(max_steps):
            # Extract block state
            block_pos = np.array(info['block_pose'][:2])
            block_angle = info['block_pose'][2]
            agent_pos = np.array(info['pos_agent'])
            goal_pos = env.goal_pose[:2]
            goal_angle = env.goal_pose[2]

            # Get action from planner
            action = geometric_planner(
                block_pos, block_angle,
                goal_pos, goal_angle,
                agent_pos, rng
            ).astype(np.float32)

            # Record data
            img = env.render(mode='rgb_array')
            state = np.concatenate([info['pos_agent'], info['block_pose']])
            keypoint = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]

            data = {
                'img': img,
                'state': np.float32(state),
                'keypoint': np.float32(keypoint),
                'action': np.float32(action),
                'n_contacts': np.float32([info['n_contacts']])
            }
            episode.append(data)

            if save_video and ep_idx < 5:
                frame = env.render(mode='rgb_array')
                # Upscale for visibility
                frame_big = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
                frames.append(frame_big)

            # Step environment
            obs, reward, done, info = env.step(action)

            if done:
                successes += 1
                break

        # Compute final reward
        final_reward = reward if not done else 1.0
        total_rewards.append(final_reward)

        # Save episode
        if len(episode) > 0:
            data_dict = {}
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')

        # Save video
        if save_video and ep_idx < 5 and len(frames) > 0:
            video_path = os.path.join(video_dir, f'episode_{ep_idx:03d}.mp4')
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f'Saved video: {video_path}')

        status = 'SUCCESS' if done else f'reward={final_reward:.3f}'
        print(f'Episode {ep_idx + 1}/{n_episodes} (seed {seed}): {status}, steps={len(episode)}')

    print(f'\n--- Summary ---')
    print(f'Total episodes: {n_episodes}')
    print(f'Successes: {successes}/{n_episodes} ({100 * successes / n_episodes:.1f}%)')
    print(f'Mean reward: {np.mean(total_rewards):.3f}')
    print(f'Data saved to: {output}')
    print(f'Total episodes in buffer: {replay_buffer.n_episodes}')


if __name__ == '__main__':
    main()

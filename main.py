from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy
import numpy as np


def main():
    # Read Video
    input_video_path = r"C:/Users/KIIT/Downloads/mini project/mini project/tennis_analysis/input_videos/Nick Kyrgios Wins First Match In Over 2 Years v s McDonald 💥 ｜ Miami 2025 Highlights [iZ8U1yEqnzs].f616.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='C:/Users/KIIT/Downloads/mini project/mini project/yolov8x.pt')
    ball_tracker = BallTracker(model_path="C:/Users/KIIT/Downloads/mini project/mini project/yolo5_last.pt")

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="C:/Users/KIIT/Downloads/mini project/mini project/tennis_analysis/tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="C:/Users/KIIT/Downloads/mini project/mini project/tennis_analysis/tracker_stubs/ball_detections.pkl"
                                                     )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    # Court Line Detector model
    court_model_path = "C:/Users/KIIT/Downloads/mini project/mini project/tennis_analysis/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    # Initialize player stats
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    }]
    
    # Filter ball shot frames that are within valid range
    max_frame = len(ball_mini_court_detections) - 1
    valid_shot_frames = [frame for frame in ball_shot_frames if frame <= max_frame]
    
    for ball_shot_ind in range(len(valid_shot_frames)-1):
        start_frame = valid_shot_frames[ball_shot_ind]
        end_frame = valid_shot_frames[ball_shot_ind+1]
        
        # Skip if frames are out of range
        if start_frame >= len(ball_mini_court_detections) or end_frame >= len(ball_mini_court_detections):
            continue
            
        # Skip if not enough ball detections in either frame
        if not ball_mini_court_detections[start_frame] or not ball_mini_court_detections[end_frame]:
            continue
            
        ball_shot_time_in_seconds = (end_frame-start_frame)/24  # 24fps

        try:
            # Get distance covered by the ball
            distance_covered_by_ball_pixels = measure_distance(
                ball_mini_court_detections[start_frame][1], 
                ball_mini_court_detections[end_frame][1]
            )

            # Convert pixel distance to meters
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
                distance_covered_by_ball_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )

            # Speed of the ball shot in km/h
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

            # Find player who hit the ball
            player_positions = player_mini_court_detections[start_frame]
            if not player_positions:
                continue
                
            player_shot_ball = min(player_positions.keys(), 
                                 key=lambda player_id: measure_distance(player_positions[player_id],
                                                                      ball_mini_court_detections[start_frame][1]))

            # Calculate opponent player speed
            opponent_player_id = 1 if player_shot_ball == 2 else 2
            
            # Skip if opponent not detected in either frame
            if opponent_player_id not in player_mini_court_detections[start_frame] or \
               opponent_player_id not in player_mini_court_detections[end_frame]:
                continue

            distance_covered_by_opponent_pixels = measure_distance(
                player_mini_court_detections[start_frame][opponent_player_id],
                player_mini_court_detections[end_frame][opponent_player_id]
            )
            
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                distance_covered_by_opponent_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )

            speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

            # Update player stats
            current_player_stats = deepcopy(player_stats_data[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
            current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

            current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

            player_stats_data.append(current_player_stats)
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            print(f"Warning: Error processing frames {start_frame}-{end_frame}: {str(e)}")
            continue

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()
    player_stats_data_df = player_stats_data_df.fillna(0)  # Fill any remaining NaN values with 0

    # Calculate averages, avoiding division by zero
    player_stats_data_df['player_1_average_shot_speed'] = np.where(
        player_stats_data_df['player_1_number_of_shots'] > 0,
        player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots'],
        0
    )
    player_stats_data_df['player_2_average_shot_speed'] = np.where(
        player_stats_data_df['player_2_number_of_shots'] > 0,
        player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots'],
        0
    )
    player_stats_data_df['player_1_average_player_speed'] = np.where(
        player_stats_data_df['player_2_number_of_shots'] > 0,
        player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots'],
        0
    )
    player_stats_data_df['player_2_average_player_speed'] = np.where(
        player_stats_data_df['player_1_number_of_shots'] > 0,
        player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots'],
        0
    )

    # Ensure index matches frame numbers
    player_stats_data_df = player_stats_data_df.set_index('frame_num')
    player_stats_data_df = player_stats_data_df.reindex(range(len(video_frames)))
    player_stats_data_df = player_stats_data_df.fillna(method='ffill').fillna(0)

    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "C:/Users/KIIT/Downloads/mini project/mini project/tennis_analysis/output_videos/output_new.avi")


if __name__ == "__main__":
    main()
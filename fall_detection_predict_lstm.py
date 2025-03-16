import os
import logging
import warnings

# 完全禁用所有警告和日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用警告
warnings.filterwarnings('ignore')
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# 禁用 absl 日志
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# 禁用所有日志输出
def disable_logging():
    logging.getLogger().setLevel(logging.ERROR)
    os.environ["PYTHONWARNINGS"] = "ignore"
    
disable_logging()

import cv2
import torch
import numpy as np
import mediapipe as mp
from fall_detection_train_lstm import FallDetectionLSTM
import time
from collections import deque

class FallDetectorLSTM:
    def __init__(self, model_path):
        try:
            self.device = torch.device('cpu')
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 初始化LSTM模型
            self.model = FallDetectionLSTM(input_size=8)  # 8个特征
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.scaler = checkpoint['scaler']
            self.sequence_length = checkpoint['sequence_length']
            
            # 初始化MediaPipe
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # 初始化特征序列缓冲区
            self.feature_buffer = deque(maxlen=self.sequence_length)
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise e
    
    def extract_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            self.bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
            
            width = x_max - x_min
            height = y_max - y_min
            
            # 计算特征
            height_width_ratio = height / width if width > 0 else 0
            bbox_area = width * height
            frame_area = h * w
            bbox_occupancy = bbox_area / frame_area
            
            features = np.array([[
                height_width_ratio,
                height_width_ratio,
                bbox_occupancy,
                width / w,
                height / h,
                height,
                h - y_max,
                (h - y_max) / height if height > 0 else 0
            ]])
            
            features = self.scaler.transform(features)
            return features, results
            
        return None, None
    
    def draw_skeleton(self, frame, results):
        if results.pose_landmarks:
            cv2.rectangle(frame, 
                         (self.bbox[0], self.bbox[1]), 
                         (self.bbox[2], self.bbox[3]), 
                         (0, 0, 255), 2)
            
            connections = self.mp_pose.POSE_CONNECTIONS
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (int(landmarks[start_idx].x * w), 
                             int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), 
                           int(landmarks[end_idx].y * h))
                
                cv2.line(frame, start_point, end_point, (255, 0, 0), 4)
            
            for landmark in landmarks:
                point = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, point, 5, (255, 0, 0), -1)
    
    def detect_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000/fps)
        
        frame_count = 0
        start_time = time.time()
        fall_detected = False
        fall_frames = 0  # 添加跌倒帧计数器
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            
            features_results = self.extract_features(frame)
            
            if features_results[0] is not None:
                features, results = features_results
                
                # 更新特征缓冲区
                self.feature_buffer.append(features[0])
                
                # 当缓冲区满时进行预测
                if len(self.feature_buffer) == self.sequence_length:
                    sequence = np.array([self.feature_buffer])
                    sequence = torch.FloatTensor(sequence).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(sequence)
                        probabilities = torch.softmax(outputs, dim=1)
                        fall_prob = probabilities[0][1].item()
                        
                        # 降低检测阈值，增加敏感度
                        if fall_prob > 0.3:  # 从0.5降到0.3
                            fall_frames += 2
                        else:
                            fall_frames = max(0, fall_frames - 1)
                        
                        if fall_frames >= 3:
                            fall_detected = True
                            cv2.putText(frame, "FALL", 
                                      (frame.shape[1]//3, frame.shape[0]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 3)
                        elif fall_frames == 0:
                            fall_detected = False
                        
                        # 显示跌倒概率（红色表示高概率）
                        prob_text = f"Fall Prob: {fall_prob:.2f}"
                        cv2.putText(frame, prob_text, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 0, 255) if fall_prob > 0.3 else (0, 255, 255), 2)
                        
                        # 显示持续时间
                        if fall_frames > 0:
                            duration_text = f"Fall Duration: {fall_frames} frames"
                            cv2.putText(frame, duration_text, (10, 90), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                self.draw_skeleton(frame, results)
            
            current_fps = frame_count / (time.time() - start_time)
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            if fall_detected:
                status = "FALL Warning" if fall_prob < 0.8 else "Falling"
            else:
                status = "Normal"
            text = f"Avg FPS: {current_fps:.1f}, Frame: {frame_count} Pred: {status}"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Fall Detection (LSTM)', frame)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\fall_detection_lstm_model.pth')
    parser.add_argument('--video', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\fall1.mp4')
    
    args = parser.parse_args()
    detector = FallDetectorLSTM(args.model)
    detector.detect_from_video(args.video) 
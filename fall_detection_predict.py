import os
import logging
import warnings

# 完全禁用所有警告和日志
warnings.filterwarnings('ignore')  # 禁用所有警告
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# 禁用 absl 日志
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# 然后再导入其他包
import cv2
import torch
import numpy as np
import mediapipe as mp
from fall_detection_train import FallDetectionModel
import time

class FallDetector:
    def __init__(self, model_path):
        try:
            self.device = torch.device('cpu')
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = FallDetectionModel(8)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.scaler = checkpoint['scaler']
            
            # 初始化 MediaPipe
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise e
    
    def extract_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 计算边界框
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 保存边界框坐标
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
            # 绘制红色边界框
            cv2.rectangle(frame, 
                         (self.bbox[0], self.bbox[1]), 
                         (self.bbox[2], self.bbox[3]), 
                         (0, 0, 255), 2)  # 红色BGR
            
            # 绘制蓝色骨骼
            connections = self.mp_pose.POSE_CONNECTIONS
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # 先绘制连接线
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (int(landmarks[start_idx].x * w), 
                             int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), 
                           int(landmarks[end_idx].y * h))
                
                cv2.line(frame, start_point, end_point, (255, 0, 0), 4)  # 增加线条粗细到4
            
            # 再绘制关键点
            for landmark in landmarks:
                point = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, point, 5, (255, 0, 0), -1)  # 增加点的大小到5
    
    def detect_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        start_time = time.time()
        fall_detected = False
        fall_frames = 0
        
        # 获取原视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算每帧应该的延迟时间（毫秒）
        delay = int(1000/fps)
        
        # 添加状态跟踪
        fall_state = {
            'start_frame': 0,
            'duration': 0,
            'is_falling': False
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            
            features_results = self.extract_features(frame)
            
            if features_results[0] is not None:
                features, results = features_results
                
                with torch.no_grad():
                    features = torch.FloatTensor(features).to(self.device)
                    outputs = self.model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    fall_prob = probabilities[0][1].item()
                    
                    # 更新跌倒状态
                    if fall_prob > 0.2:
                        if not fall_state['is_falling']:
                            fall_state['start_frame'] = frame_count
                            fall_state['is_falling'] = True
                        fall_state['duration'] = frame_count - fall_state['start_frame']
                        
                        if fall_state['duration'] >= 2:  # 至少持续2帧
                            fall_detected = True
                            # 添加持续时间显示
                            cv2.putText(frame, f"Fall Duration: {fall_state['duration']} frames", 
                                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        fall_state['is_falling'] = False
                        fall_state['duration'] = 0
                        fall_detected = False
                    
                    # 显示更详细的状态信息
                    status_text = f"Status: {'Falling' if fall_state['is_falling'] else 'Normal'}"
                    cv2.putText(frame, status_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                self.draw_skeleton(frame, results)
                
                prob_color = (0, 0, 255) if fall_prob > 0.2 else (0, 255, 255)
                prob_text = f"Fall Prob: {fall_prob:.2f}"
                cv2.putText(frame, prob_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, prob_color, 2)
            
            current_fps = frame_count / (time.time() - start_time)
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            if fall_detected:
                status = "FALL Warning" if fall_prob < 0.8 else "Falling"
            else:
                status = "Normal"
            
            text = f"Avg FPS: {current_fps:.1f}, Frame: {frame_count} Pred: {status}"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Fall Detection', frame)
            
            # 使用原视频的帧率来控制播放速度
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, 
                           default=r'D:\xiaoD_keshe\kes\dataset\fall_detection_model.pth')
        parser.add_argument('--video', type=str, 
                           default=r'D:\xiaoD_keshe\kes\dataset\fall1.mp4')
        
        args = parser.parse_args()
        
        # 检查文件是否存在
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"找不到模型文件: {args.model}")
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"找不到视频文件: {args.video}")
            
        detector = FallDetector(args.model)
        detector.detect_from_video(args.video)
    except Exception as e:
        print(f"程序出错: {str(e)}")
        raise e 
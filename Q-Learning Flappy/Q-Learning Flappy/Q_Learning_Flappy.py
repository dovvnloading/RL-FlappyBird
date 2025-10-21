import sys
import time
import random
import collections
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QPlainTextEdit,
                               QProgressBar, QFormLayout, QSplitter, QGroupBox,
                               QDoubleSpinBox, QSpinBox, QGridLayout)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPoint
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QLinearGradient, QRadialGradient, 
                           QIcon, QPixmap, QFont)

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_X = 50
BIRD_RADIUS = 20
PIPE_WIDTH = 70
PIPE_GAP = 200
GROUND_HEIGHT = 100

# Default Learning Parameters (will be controlled by UI)
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_EXPLORATION_DECAY = 0.99995
DEFAULT_MIN_EXPLORATION = 0.01
DEFAULT_TRAINING_EPISODES = 20000

def create_app_icon():
    """Programmatically creates a blue 3D square icon."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    face_color = QColor(50, 150, 255)
    top_color = face_color.lighter(120)
    side_color = face_color.darker(120)
    origin = QPoint(32, 32)
    front_face = [origin + QPoint(-18, -18), origin + QPoint(18, -18), origin + QPoint(18, 18), origin + QPoint(-18, 18)]
    top_face = [origin + QPoint(-18, -18), origin + QPoint(0, -28), origin + QPoint(36, -28), origin + QPoint(18, -18)]
    side_face = [origin + QPoint(18, -18), origin + QPoint(36, -28), origin + QPoint(36, 8), origin + QPoint(18, 18)]
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(top_color)); painter.drawPolygon(top_face)
    painter.setBrush(QBrush(side_color)); painter.drawPolygon(side_face)
    painter.setBrush(QBrush(face_color)); painter.drawPolygon(front_face)
    painter.end()
    return QIcon(pixmap)


class FlappyBirdEnvironment:
    """Represents the Flappy Bird game environment."""
    def __init__(self):
        self.bird_y = SCREEN_HEIGHT // 2; self.bird_velocity = 0
        self.pipes = [self._generate_pipe()]; self.score = 0; self.alive = True
    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2; self.bird_velocity = 0
        self.pipes = [self._generate_pipe()]; self.score = 0; self.alive = True
        return self._get_state()
    def _generate_pipe(self):
        opening_y = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
        return {'x': SCREEN_WIDTH, 'opening_y': opening_y}
    def _get_state(self):
        if not self.pipes: return None
        next_pipe = self.pipes[0]
        return (int(self.bird_y / 50), int(self.bird_velocity / 2),
                int((next_pipe['x'] - BIRD_X) / 50), int(next_pipe['opening_y'] / 50))
    def step(self, action):
        if action == 1: self.bird_velocity = -6
        self.bird_velocity += 0.5; self.bird_y += self.bird_velocity
        for pipe in self.pipes: pipe['x'] -= 5
        self.pipes = [p for p in self.pipes if p['x'] > -PIPE_WIDTH]
        if not self.pipes or self.pipes[-1]['x'] < SCREEN_WIDTH - 200:
            self.pipes.append(self._generate_pipe())
        reward = 1
        if self.bird_y <= 0 or self.bird_y >= SCREEN_HEIGHT - GROUND_HEIGHT:
            self.alive = False; reward = -100
            return self._get_state(), reward, not self.alive
        for pipe in self.pipes:
            # **FIXED**: Corrected collision detection logic
            if pipe['x'] < BIRD_X + BIRD_RADIUS and pipe['x'] + PIPE_WIDTH > BIRD_X - BIRD_RADIUS:
                if self.bird_y < pipe['opening_y'] or self.bird_y > pipe['opening_y'] + PIPE_GAP:
                    self.alive = False; reward = -100
                    return self._get_state(), reward, not self.alive
                # This part of the reward logic seems tricky, but we leave it as is.
                if pipe['x'] + PIPE_WIDTH < BIRD_X:
                    reward = 50; self.score += 1
        return self._get_state(), reward, not self.alive


class QLearningAgent:
    """Implements the Q-learning agent that learns to play Flappy Bird."""
    def __init__(self, learning_rate, discount_factor, exploration_decay, min_exploration, action_space=2):
        self.q_table = {}; self.learning_rate = learning_rate; self.discount_factor = discount_factor
        self.exploration_rate = 1.0; self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration; self.action_space = action_space
    def get_action(self, state):
        if random.random() < self.exploration_rate or state not in self.q_table:
            return random.randint(0, self.action_space - 1)
        return max(range(self.action_space), key=lambda a: self.q_table[state][a])
    def update(self, state, action, reward, next_state, done):
        if state is None or next_state is None: return
        if state not in self.q_table: self.q_table[state] = [0, 0]
        if next_state not in self.q_table: self.q_table[next_state] = [0, 0]
        current_q = self.q_table[state][action]
        max_next_q = 0 if done else max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)


class TrainingThread(QThread):
    """A QThread for running the training process in a separate thread."""
    stats_updated = Signal(dict)
    training_complete = Signal(object)

    def __init__(self, episodes, initial_lr, final_lr, lr_decay_steps, discount_factor, exploration_decay, min_exploration):
        super().__init__()
        self.episodes = episodes; self.initial_lr = initial_lr
        self.final_lr = final_lr; self.lr_decay_steps = lr_decay_steps
        self.discount_factor = discount_factor; self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

    def run(self):
        env = FlappyBirdEnvironment()
        agent = QLearningAgent(
            learning_rate=self.initial_lr, discount_factor=self.discount_factor,
            exploration_decay=self.exploration_decay, min_exploration=self.min_exploration
        )
        
        lr_decay_factor = 1.0
        if self.lr_decay_steps > 0 and self.initial_lr > self.final_lr:
            lr_decay_factor = (self.final_lr / self.initial_lr)**(1 / self.lr_decay_steps)

        episodes_per_decay_step = self.episodes // self.lr_decay_steps if self.lr_decay_steps > 0 else self.episodes

        scores = collections.deque(maxlen=100); start_time = time.time(); last_update_time = 0
        for episode in range(self.episodes):
            if self.isInterruptionRequested():
                print("Training interrupted by user."); break
            
            if self.lr_decay_steps > 0 and episode > 0 and episode % episodes_per_decay_step == 0:
                new_lr = max(self.final_lr, agent.learning_rate * lr_decay_factor)
                agent.learning_rate = new_lr

            state = env.reset(); done = False
            while not done:
                action = agent.get_action(state); next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done); state = next_state
            scores.append(env.score)
            current_time = time.time()
            if current_time - last_update_time > 0.5:
                elapsed_time = current_time - start_time
                episodes_per_second = episode / elapsed_time if elapsed_time > 0 else 0
                remaining_episodes = self.episodes - episode
                etr = remaining_episodes / episodes_per_second if episodes_per_second > 0 else 0
                stats = {"episode": episode, "max_score": max(scores) if scores else 0,
                         "avg_score": np.mean(scores) if scores else 0, "exploration_rate": agent.exploration_rate,
                         "learning_rate": agent.learning_rate,
                         "q_table_size": len(agent.q_table), "elapsed_time": elapsed_time, "etr": etr}
                self.stats_updated.emit(stats); last_update_time = current_time
        self.training_complete.emit(agent)


class FlappyBirdTrainingApp(QMainWindow):
    """The main application window with an improved UI for monitoring training."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flappy Bird Q-Learning Dashboard")
        self.setWindowIcon(create_app_icon()); self.setGeometry(100, 100, 950, 700)

        self.episodes = []; self.avg_scores = []; self.max_scores = []

        main_widget = QWidget(); main_layout = QHBoxLayout(main_widget); self.setCentralWidget(main_widget)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel); left_panel.setMaximumWidth(320)

        control_layout = QGridLayout()
        self.start_btn = QPushButton("Start Training"); self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("Stop Training"); self.stop_btn.clicked.connect(self.stop_training); self.stop_btn.setEnabled(False)
        self.play_btn = QPushButton("Play Game"); self.play_btn.clicked.connect(self.play_game)
        self.demo_btn = QPushButton("Demonstrate"); self.demo_btn.clicked.connect(self.demonstrate); self.demo_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn, 0, 0)
        control_layout.addWidget(self.stop_btn, 0, 1)
        control_layout.addWidget(self.play_btn, 1, 0)
        control_layout.addWidget(self.demo_btn, 1, 1)

        params_groupbox = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        self.episodes_spinbox = QSpinBox(); self.episodes_spinbox.setRange(1000, 1000000); self.episodes_spinbox.setValue(DEFAULT_TRAINING_EPISODES); self.episodes_spinbox.setSingleStep(1000)
        
        self.initial_lr_spinbox = QDoubleSpinBox(); self.initial_lr_spinbox.setRange(0.001, 1.0); self.initial_lr_spinbox.setValue(0.5); self.initial_lr_spinbox.setSingleStep(0.01); self.initial_lr_spinbox.setDecimals(3)
        self.final_lr_spinbox = QDoubleSpinBox(); self.final_lr_spinbox.setRange(0.001, 1.0); self.final_lr_spinbox.setValue(0.01); self.final_lr_spinbox.setSingleStep(0.01); self.final_lr_spinbox.setDecimals(3)
        self.lr_decay_steps_spinbox = QSpinBox(); self.lr_decay_steps_spinbox.setRange(1, 100); self.lr_decay_steps_spinbox.setValue(5)

        self.discount_spinbox = QDoubleSpinBox(); self.discount_spinbox.setRange(0.8, 0.999); self.discount_spinbox.setValue(DEFAULT_DISCOUNT_FACTOR); self.discount_spinbox.setSingleStep(0.01); self.discount_spinbox.setDecimals(3)
        self.decay_spinbox = QDoubleSpinBox(); self.decay_spinbox.setRange(0.99, 0.99999); self.decay_spinbox.setValue(DEFAULT_EXPLORATION_DECAY); self.decay_spinbox.setSingleStep(0.00001); self.decay_spinbox.setDecimals(5)
        
        params_layout.addRow("Episodes:", self.episodes_spinbox)
        params_layout.addRow("Initial Learning Rate:", self.initial_lr_spinbox)
        params_layout.addRow("Final Learning Rate:", self.final_lr_spinbox)
        params_layout.addRow("LR Decay Steps:", self.lr_decay_steps_spinbox)
        params_layout.addRow("Discount Factor (γ):", self.discount_spinbox)
        params_layout.addRow("Exploration Decay:", self.decay_spinbox)
        params_groupbox.setLayout(params_layout)

        stats_groupbox = QGroupBox("Live Statistics")
        stats_form_layout = QFormLayout()
        self.progress_bar = QProgressBar()
        self.episode_label = QLabel("N/A"); self.avg_score_label = QLabel("N/A"); self.max_score_label = QLabel("N/A")
        self.exploration_label = QLabel("N/A"); self.lr_label = QLabel("N/A"); self.q_table_label = QLabel("N/A"); self.elapsed_time_label = QLabel("N/A"); self.etr_label = QLabel("N/A")
        stats_form_layout.addRow("Progress:", self.progress_bar)
        stats_form_layout.addRow("Episode:", self.episode_label)
        stats_form_layout.addRow("Avg Score (last 100):", self.avg_score_label)
        stats_form_layout.addRow("Max Score (last 100):", self.max_score_label)
        stats_form_layout.addRow("Exploration Rate:", self.exploration_label)
        stats_form_layout.addRow("Current Learning Rate:", self.lr_label)
        stats_form_layout.addRow("Q-Table Size:", self.q_table_label)
        stats_form_layout.addRow("Elapsed Time:", self.elapsed_time_label)
        stats_form_layout.addRow("Est. Time Remaining:", self.etr_label)
        stats_groupbox.setLayout(stats_form_layout)

        left_layout.addLayout(control_layout); left_layout.addWidget(params_groupbox); left_layout.addWidget(stats_groupbox); left_layout.addStretch()

        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel); splitter = QSplitter(Qt.Orientation.Vertical)

        self.plot_widget = pg.PlotWidget()
        
        # --- Custom Graph Styling ---
        BG_COLOR = QColor(25, 35, 45); AXIS_COLOR = QColor(200, 200, 200)
        AVG_SCORE_COLOR = QColor(0, 170, 255); MAX_SCORE_COLOR = QColor(255, 165, 0)
        self.plot_widget.setBackground(BG_COLOR); self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=AXIS_COLOR)); self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color=AXIS_COLOR))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=AXIS_COLOR)); self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=AXIS_COLOR))
        self.plot_widget.setLabel('left', 'Score', color=str(AXIS_COLOR.name()), fontSize='11pt'); self.plot_widget.setLabel('bottom', 'Episode', color=str(AXIS_COLOR.name()), fontSize='11pt')
        legend = self.plot_widget.addLegend(); legend.setBrush(QColor(50, 50, 50, 150)); legend.setLabelTextColor(AXIS_COLOR)
        avg_gradient = QLinearGradient(0, 0, 0, 300)
        avg_gradient_start_color = QColor(AVG_SCORE_COLOR); avg_gradient_start_color.setAlpha(150)
        avg_gradient_end_color = QColor(AVG_SCORE_COLOR); avg_gradient_end_color.setAlpha(0)
        avg_gradient.setColorAt(0.0, avg_gradient_start_color); avg_gradient.setColorAt(1.0, avg_gradient_end_color)
        avg_brush = QBrush(avg_gradient); avg_pen = pg.mkPen(color=AVG_SCORE_COLOR, width=2.5)
        max_pen = pg.mkPen(color=MAX_SCORE_COLOR, width=2); max_shadow_pen = pg.mkPen(color=QColor(0,0,0,70), width=6, cosmetic=True)
        self.avg_score_plot = self.plot_widget.plot(pen=avg_pen, name="Average Score", fillLevel=0, brush=avg_brush)
        self.max_score_plot = self.plot_widget.plot(pen=max_pen, name="Max Score", shadowPen=max_shadow_pen)
        # --- End of Custom Graph Styling ---

        self.log_output = QPlainTextEdit(); self.log_output.setReadOnly(True)
        splitter.addWidget(self.plot_widget); splitter.addWidget(self.log_output); splitter.setSizes([400, 200])
        right_layout.addWidget(splitter)
        main_layout.addWidget(left_panel); main_layout.addWidget(right_panel, stretch=1)
        self.training_thread = None; self.trained_agent = None

    def start_training(self):
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.demo_btn.setEnabled(False)
        self.log_output.clear(); self.episodes, self.avg_scores, self.max_scores = [], [], []
        
        episodes = self.episodes_spinbox.value()
        initial_lr = self.initial_lr_spinbox.value()
        final_lr = self.final_lr_spinbox.value()
        lr_decay_steps = self.lr_decay_steps_spinbox.value()
        discount = self.discount_spinbox.value()
        decay = self.decay_spinbox.value()

        self.log("Starting new training session with adaptive learning rate...")
        self.log(f"Params: Episodes={episodes}, Initial LR={initial_lr}, Final LR={final_lr}, Decay Steps={lr_decay_steps}")

        self.training_thread = TrainingThread(episodes=episodes, initial_lr=initial_lr, final_lr=final_lr,
                                              lr_decay_steps=lr_decay_steps, discount_factor=discount,
                                              exploration_decay=decay, min_exploration=DEFAULT_MIN_EXPLORATION)
        self.training_thread.stats_updated.connect(self.update_training_ui); self.training_thread.training_complete.connect(self.training_finished)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.requestInterruption(); self.stop_btn.setEnabled(False)
            self.log("Stop request sent. Waiting for thread to finish current episode...")

    def update_training_ui(self, stats):
        episode = stats["episode"]; total_episodes = self.training_thread.episodes
        self.progress_bar.setValue(int(100 * episode / total_episodes)); self.episode_label.setText(f"{episode} / {total_episodes}")
        self.avg_score_label.setText(f"{stats['avg_score']:.2f}"); self.max_score_label.setText(f"{stats['max_score']:.2f}")
        self.exploration_label.setText(f"{stats['exploration_rate']:.4f}"); self.lr_label.setText(f"{stats['learning_rate']:.4f}")
        self.q_table_label.setText(f"{stats['q_table_size']:,}")
        elapsed_m, elapsed_s = divmod(int(stats['elapsed_time']), 60); etr_m, etr_s = divmod(int(stats['etr']), 60)
        self.elapsed_time_label.setText(f"{elapsed_m:02d}:{elapsed_s:02d}"); self.etr_label.setText(f"{etr_m:02d}:{etr_s:02d}")
        self.episodes.append(episode); self.avg_scores.append(stats['avg_score']); self.max_scores.append(stats['max_score'])
        self.avg_score_plot.setData(self.episodes, self.avg_scores); self.max_score_plot.setData(self.episodes, self.max_scores)
        if episode > 0 and episode % 500 == 0:
            self.log(f"Ep {episode}: Avg Score={stats['avg_score']:.2f}, LR={stats['learning_rate']:.4f}, Exp={stats['exploration_rate']:.3f}")

    def training_finished(self, agent):
        self.trained_agent = agent; self.log("\nTraining finished!")
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.demo_btn.setEnabled(True)
        self.progress_bar.setValue(100)

    def demonstrate(self):
        if self.trained_agent: self.demo_window = FlappyBirdDemoWindow(self.trained_agent); self.demo_window.show()
        else: self.log("No trained agent available to demonstrate.")

    def play_game(self):
        self.play_window = FlappyBirdPlayWindow()
        self.play_window.show()

    def log(self, message): self.log_output.appendPlainText(message)
    def closeEvent(self, event):
        self.stop_training()
        if self.training_thread: self.training_thread.wait()
        event.accept()


class FlappyBirdDemoWindow(QMainWindow):
    """A window to demonstrate the trained Flappy Bird agent."""
    def __init__(self, agent):
        super().__init__()
        self.setWindowTitle("Flappy Bird Demo"); self.setGeometry(100, 100, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setWindowIcon(create_app_icon()); self.agent = agent; self.agent.exploration_rate = 0
        self.env = FlappyBirdEnvironment()
        self.bird_wing_angle = 0; self.bird_rotation = 0
        self.cloud_positions = [{'x': random.randint(0, SCREEN_WIDTH), 'y': random.randint(50, 200)} for _ in range(3)]
        self.ground_offset = 0; self.state = self.env.reset()
        self.game_over = False; self.high_score = 0
        self.game_timer = QTimer(self); self.game_timer.timeout.connect(self.update_game); self.game_timer.start(1000 // 60)

    def update_game(self):
        if not self.game_over:
            action = self.agent.get_action(self.state); self.state, _, done = self.env.step(action)
            if self.env.score > self.high_score: self.high_score = self.env.score
            self.bird_wing_angle = 10 * np.sin(time.time() * 15); self.bird_rotation = np.clip(self.env.bird_velocity * 3, -30, 30)
            for cloud in self.cloud_positions:
                cloud['x'] -= 1
                if cloud['x'] < -100: cloud['x'] = SCREEN_WIDTH + 50; cloud['y'] = random.randint(50, 200)
            self.ground_offset += 5
            if done: self.game_over = True
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, SCREEN_HEIGHT); gradient.setColorAt(0, QColor(135, 206, 235)); gradient.setColorAt(0.7, QColor(200, 230, 255)); painter.fillRect(self.rect(), gradient)
        painter.setBrush(QBrush(QColor(255, 255, 255, 200))); painter.setPen(Qt.PenStyle.NoPen)
        for cloud in self.cloud_positions:
            cx, cy = cloud['x'], cloud['y']; painter.drawEllipse(cx - 20, cy, 40, 30); painter.drawEllipse(cx, cy - 10, 50, 35); painter.drawEllipse(cx + 20, cy, 40, 30)
        for pipe in self.env.pipes:
            pipe_gradient = QLinearGradient(pipe['x'], 0, pipe['x'] + PIPE_WIDTH, 0)
            pipe_gradient.setColorAt(0, QColor(40, 180, 40)); pipe_gradient.setColorAt(0.5, QColor(0, 200, 0)); pipe_gradient.setColorAt(1, QColor(40, 180, 40))
            painter.setBrush(QBrush(pipe_gradient)); painter.setPen(QPen(QColor(30, 130, 30), 2))
            pipe_top = pipe['opening_y']; painter.drawRect(pipe['x'], 0, PIPE_WIDTH, pipe_top); painter.drawRect(pipe['x'] - 5, pipe_top - 20, PIPE_WIDTH + 10, 20)
            pipe_bottom = pipe['opening_y'] + PIPE_GAP; painter.drawRect(pipe['x'], pipe_bottom, PIPE_WIDTH, SCREEN_HEIGHT - pipe_bottom); painter.drawRect(pipe['x'] - 5, pipe_bottom, PIPE_WIDTH + 10, 20)
        painter.save(); painter.translate(BIRD_X, int(self.env.bird_y)); painter.rotate(self.bird_rotation)
        bird_gradient = QRadialGradient(0, 0, BIRD_RADIUS); bird_gradient.setColorAt(0, QColor(255, 220, 0)); bird_gradient.setColorAt(1, QColor(255, 180, 0))
        painter.setBrush(QBrush(bird_gradient)); painter.setPen(QPen(QColor(200, 150, 0), 2)); painter.drawEllipse(-BIRD_RADIUS, -BIRD_RADIUS, BIRD_RADIUS * 2, BIRD_RADIUS * 2)
        wing_points = [QPoint(-5, -5), QPoint(-20, int(-5 + self.bird_wing_angle)), QPoint(-5, 5)]; painter.drawPolygon(wing_points)
        painter.setBrush(QBrush(Qt.GlobalColor.white)); painter.drawEllipse(5, -8, 8, 8); painter.setBrush(QBrush(Qt.GlobalColor.black)); painter.drawEllipse(8, -6, 4, 4)
        beak_points = [QPoint(BIRD_RADIUS - 2, -2), QPoint(BIRD_RADIUS + 10, 0), QPoint(BIRD_RADIUS - 2, 2)]; painter.setBrush(QBrush(QColor(255, 150, 0))); painter.drawPolygon(beak_points)
        painter.restore()
        ground_gradient = QLinearGradient(0, SCREEN_HEIGHT - GROUND_HEIGHT, 0, SCREEN_HEIGHT); ground_gradient.setColorAt(0, QColor(139, 100, 19)); ground_gradient.setColorAt(1, QColor(100, 50, 0))
        painter.setBrush(QBrush(ground_gradient)); painter.setPen(Qt.PenStyle.NoPen); painter.drawRect(0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT)
        painter.setPen(QPen(QColor(120, 80, 10), 2))
        for x in range(-int(self.ground_offset) % 40, SCREEN_WIDTH, 40): painter.drawLine(x, SCREEN_HEIGHT - GROUND_HEIGHT + 10, x + 20, SCREEN_HEIGHT - GROUND_HEIGHT + 5)
        
        self.draw_hud(painter)

        if self.game_over:
            painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            painter.drawRect(self.rect())
            
            painter.setPen(QPen(QColor(255, 255, 255)))
            
            font_large = QFont("Arial", 40, QFont.Weight.Bold)
            painter.setFont(font_large)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Game Over")
            
            font_small = QFont("Arial", 16)
            painter.setFont(font_small)
            painter.drawText(self.rect().translated(0, 50), Qt.AlignmentFlag.AlignCenter, "Click to Replay")

    def draw_hud(self, painter):
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)
        
        score_text = f"Score: {self.env.score}"
        high_score_text = f"High: {self.high_score}"
        
        painter.setPen(QPen(QColor(0, 0, 0, 150)))
        painter.drawText(12, 37, score_text)
        painter.drawText(self.rect().adjusted(0, 7, -12, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, high_score_text)
        
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(10, 35, score_text)
        painter.drawText(self.rect().adjusted(0, 5, -10, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, high_score_text)

    def mousePressEvent(self, event):
        if self.game_over:
            self.state = self.env.reset()
            self.game_over = False


class FlappyBirdPlayWindow(QMainWindow):
    """A window for a human user to play Flappy Bird."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flappy Bird - Play"); self.setGeometry(100, 100, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setWindowIcon(create_app_icon())
        self.env = FlappyBirdEnvironment()
        self.bird_wing_angle = 0; self.bird_rotation = 0
        self.cloud_positions = [{'x': random.randint(0, SCREEN_WIDTH), 'y': random.randint(50, 200)} for _ in range(3)]
        self.ground_offset = 0; self.env.reset()
        self.game_over = False; self.high_score = 0; self.flap_pending = False
        self.game_timer = QTimer(self); self.game_timer.timeout.connect(self.update_game); self.game_timer.start(1000 // 60)

    def update_game(self):
        if not self.game_over:
            action = 1 if self.flap_pending else 0
            self.flap_pending = False
            _, _, done = self.env.step(action)
            if self.env.score > self.high_score: self.high_score = self.env.score
            self.bird_wing_angle = 10 * np.sin(time.time() * 15); self.bird_rotation = np.clip(self.env.bird_velocity * 3, -30, 30)
            for cloud in self.cloud_positions:
                cloud['x'] -= 1
                if cloud['x'] < -100: cloud['x'] = SCREEN_WIDTH + 50; cloud['y'] = random.randint(50, 200)
            self.ground_offset += 5
            if done: self.game_over = True
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, SCREEN_HEIGHT); gradient.setColorAt(0, QColor(135, 206, 235)); gradient.setColorAt(0.7, QColor(200, 230, 255)); painter.fillRect(self.rect(), gradient)
        painter.setBrush(QBrush(QColor(255, 255, 255, 200))); painter.setPen(Qt.PenStyle.NoPen)
        for cloud in self.cloud_positions:
            cx, cy = cloud['x'], cloud['y']; painter.drawEllipse(cx - 20, cy, 40, 30); painter.drawEllipse(cx, cy - 10, 50, 35); painter.drawEllipse(cx + 20, cy, 40, 30)
        for pipe in self.env.pipes:
            pipe_gradient = QLinearGradient(pipe['x'], 0, pipe['x'] + PIPE_WIDTH, 0)
            pipe_gradient.setColorAt(0, QColor(40, 180, 40)); pipe_gradient.setColorAt(0.5, QColor(0, 200, 0)); pipe_gradient.setColorAt(1, QColor(40, 180, 40))
            painter.setBrush(QBrush(pipe_gradient)); painter.setPen(QPen(QColor(30, 130, 30), 2))
            pipe_top = pipe['opening_y']; painter.drawRect(pipe['x'], 0, PIPE_WIDTH, pipe_top); painter.drawRect(pipe['x'] - 5, pipe_top - 20, PIPE_WIDTH + 10, 20)
            pipe_bottom = pipe['opening_y'] + PIPE_GAP; painter.drawRect(pipe['x'], pipe_bottom, PIPE_WIDTH, SCREEN_HEIGHT - pipe_bottom); painter.drawRect(pipe['x'] - 5, pipe_bottom, PIPE_WIDTH + 10, 20)
        painter.save(); painter.translate(BIRD_X, int(self.env.bird_y)); painter.rotate(self.bird_rotation)
        bird_gradient = QRadialGradient(0, 0, BIRD_RADIUS); bird_gradient.setColorAt(0, QColor(255, 220, 0)); bird_gradient.setColorAt(1, QColor(255, 180, 0))
        painter.setBrush(QBrush(bird_gradient)); painter.setPen(QPen(QColor(200, 150, 0), 2)); painter.drawEllipse(-BIRD_RADIUS, -BIRD_RADIUS, BIRD_RADIUS * 2, BIRD_RADIUS * 2)
        wing_points = [QPoint(-5, -5), QPoint(-20, int(-5 + self.bird_wing_angle)), QPoint(-5, 5)]; painter.drawPolygon(wing_points)
        painter.setBrush(QBrush(Qt.GlobalColor.white)); painter.drawEllipse(5, -8, 8, 8); painter.setBrush(QBrush(Qt.GlobalColor.black)); painter.drawEllipse(8, -6, 4, 4)
        beak_points = [QPoint(BIRD_RADIUS - 2, -2), QPoint(BIRD_RADIUS + 10, 0), QPoint(BIRD_RADIUS - 2, 2)]; painter.setBrush(QBrush(QColor(255, 150, 0))); painter.drawPolygon(beak_points)
        painter.restore()
        ground_gradient = QLinearGradient(0, SCREEN_HEIGHT - GROUND_HEIGHT, 0, SCREEN_HEIGHT); ground_gradient.setColorAt(0, QColor(139, 100, 19)); ground_gradient.setColorAt(1, QColor(100, 50, 0))
        painter.setBrush(QBrush(ground_gradient)); painter.setPen(Qt.PenStyle.NoPen); painter.drawRect(0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT)
        painter.setPen(QPen(QColor(120, 80, 10), 2))
        for x in range(-int(self.ground_offset) % 40, SCREEN_WIDTH, 40): painter.drawLine(x, SCREEN_HEIGHT - GROUND_HEIGHT + 10, x + 20, SCREEN_HEIGHT - GROUND_HEIGHT + 5)

        self.draw_hud(painter)

        if self.game_over:
            painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            painter.drawRect(self.rect())
            painter.setPen(QPen(QColor(255, 255, 255)))
            font_large = QFont("Arial", 40, QFont.Weight.Bold)
            painter.setFont(font_large)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Game Over")
            font_small = QFont("Arial", 16)
            painter.setFont(font_small)
            painter.drawText(self.rect().translated(0, 50), Qt.AlignmentFlag.AlignCenter, "Click to Replay")

    def draw_hud(self, painter):
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)
        score_text = f"Score: {self.env.score}"
        high_score_text = f"High: {self.high_score}"
        painter.setPen(QPen(QColor(0, 0, 0, 150)))
        painter.drawText(12, 37, score_text)
        painter.drawText(self.rect().adjusted(0, 7, -12, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, high_score_text)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(10, 35, score_text)
        painter.drawText(self.rect().adjusted(0, 5, -10, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, high_score_text)

    def mousePressEvent(self, event):
        if self.game_over:
            self.env.reset()
            self.game_over = False

    def keyPressEvent(self, event):
        if not self.game_over and event.key() == Qt.Key.Key_Space:
            self.flap_pending = True


def main():
    app = QApplication(sys.argv)
    main_window = FlappyBirdTrainingApp()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
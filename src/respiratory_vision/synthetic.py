from pathlib import Path
import json

import cv2
import numpy as np


def generate_breathing_video(
    output_path: str | Path,
    bpm: float = 18.0,
    duration_seconds: float = 20.0,
    fps: int = 10,
    width: int = 200,
    height: int = 150,
    amplitude_pixels: float = 5.0,
    seed: int = 7,
) -> dict:
    """Generate a non-clinical synthetic torso-motion video with known ground truth."""
    if bpm <= 0 or duration_seconds <= 0 or fps <= 1:
        raise ValueError("bpm, duration_seconds, and fps must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Unable to initialize MP4 writer.")

    # Fixed textured torso canvas. Texture ensures optical-flow features.
    torso_w, torso_h = int(width * 0.54), int(height * 0.42)
    torso_x = (width - torso_w) // 2
    torso_y0 = int(height * 0.42)
    texture = rng.integers(70, 190, size=(torso_h, torso_w), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    texture_bgr = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    n_frames = int(round(duration_seconds * fps))
    hz = bpm / 60.0

    for i in range(n_frames):
        t = i / fps
        # Slowly changing background avoids an unrealistically static scene.
        base = 25 + 3 * np.sin(2 * np.pi * 0.03 * t)
        frame = np.full((height, width, 3), base, dtype=np.uint8)

        dy = int(round(amplitude_pixels * np.sin(2 * np.pi * hz * t)))
        torso_y = torso_y0 + dy

        # Add a subtle chest-like shape and textured region.
        y1, y2 = torso_y, torso_y + torso_h
        x1, x2 = torso_x, torso_x + torso_w
        if 0 <= y1 and y2 <= height:
            frame[y1:y2, x1:x2] = texture_bgr

        # Shoulder/chest boundary gives stable gradients.
        cv2.ellipse(
            frame,
            (width // 2, torso_y + torso_h // 3),
            (torso_w // 2, torso_h // 3),
            0, 180, 360,
            (205, 205, 205),
            2,
        )
        writer.write(frame)

    writer.release()

    metadata = {
        "synthetic": True,
        "clinical": False,
        "ground_truth_bpm": float(bpm),
        "duration_seconds": float(duration_seconds),
        "fps": int(fps),
        "width": int(width),
        "height": int(height),
        "motion_amplitude_pixels": float(amplitude_pixels),
        "seed": int(seed),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


from dataclasses import dataclass, asdict

CONDITIONS = ('clean','mild_noise','camera_motion','low_amplitude','illumination_changes','occlusion','irregular_breathing','combined_stress')

@dataclass(frozen=True)
class ChallengeConfig:
    bpm: float = 18.
    duration_seconds: float = 24.
    fps: float = 10.
    width: int = 200
    height: int = 150
    amplitude_pixels: float = 4.
    phase: float = 0.
    noise_std: float = 0.
    camera_amplitude: float = 0.
    camera_hz: float = .23
    illumination_amplitude: float = 0.
    occlusion_fraction: float = 0.
    artifact_pixels: float = 0.
    rate_change_bpm: float = 0.
    irregularity: float = 0.
    pause_fraction: float = 0.
    seed: int = 11
    condition: str = 'clean'

    def validate(self):
        numbers = [v for v in asdict(self).values() if isinstance(v,(float,int))]
        if not np.all(np.isfinite(numbers)) or self.bpm<=0 or self.duration_seconds<=0 or self.fps<=1 or self.width<64 or self.height<64:
            raise ValueError('Invalid challenge dimensions, rate, duration or frame rate.')
        if self.amplitude_pixels<0 or self.noise_std<0 or self.camera_amplitude<0 or self.camera_hz<=0 or self.artifact_pixels<0 or self.illumination_amplitude<0:
            raise ValueError('Artifact magnitudes must be nonnegative.')
        if not 0<=self.occlusion_fraction<.9 or not 0<=self.pause_fraction<.8 or not 0<=self.irregularity<.8 or self.bpm+min(0,self.rate_change_bpm)<=0:
            raise ValueError('Invalid occlusion, pause, irregularity or rate change.')

def challenge_config(condition: str, split='development', index=0) -> ChallengeConfig:
    if condition not in CONDITIONS or split not in ('development','test') or index<0:
        raise ValueError('Unknown condition/split or negative index.')
    test = split=='test'
    base = dict(bpm=([13.,19.,25.,31.] if test else [12.,18.,24.,30.])[index%4],
                fps=12. if test else 10., phase=.71 if test else .13,
                seed=(51000 if test else 11000)+CONDITIONS.index(condition)*100+index,
                condition=condition)
    severity=1.2 if test else 1.
    changes = {
        'clean':{},'mild_noise':{'noise_std':4*severity},
        'camera_motion':{'camera_amplitude':7*severity},
        'low_amplitude':{'amplitude_pixels':.025*severity},
        'illumination_changes':{'illumination_amplitude':35*severity},
        'occlusion':{'occlusion_fraction':.45},
        'irregular_breathing':{'irregularity':.3,'rate_change_bpm':6*severity,'pause_fraction':.08},
        'combined_stress':{'noise_std':7*severity,'camera_amplitude':9*severity,'illumination_amplitude':30.,'occlusion_fraction':.35,'artifact_pixels':12.,'irregularity':.2}}
    return ChallengeConfig(**(base|changes[condition]))

def generate_challenge_video(output_path: str | Path, config: ChallengeConfig) -> dict:
    """Deterministic engineering stress cases; does not simulate clinical physiology."""
    config.validate()
    c=config
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(c.seed)
    count=round(c.duration_seconds*c.fps)
    t=np.arange(count)/c.fps
    instantaneous=c.bpm+c.rate_change_bpm*t/c.duration_seconds+c.bpm*c.irregularity*np.sin(2*np.pi*.073*t+c.phase)
    pause=(t>c.duration_seconds*.45)&(t<c.duration_seconds*(.45+c.pause_fraction)) if c.pause_fraction else np.zeros(count,dtype=bool)
    instantaneous[pause]=0
    phase=c.phase+2*np.pi*np.r_[0,np.cumsum(instantaneous[:-1])]/(60*c.fps)
    displacement=c.amplitude_pixels*np.sin(phase)
    background=cv2.GaussianBlur(rng.integers(20,90,(c.height,c.width,3),dtype=np.uint8),(3,3),0)
    torso=np.zeros_like(background)
    mask=np.zeros((c.height,c.width),np.uint8)
    x,y,w,h=int(c.width*.23),int(c.height*.40),int(c.width*.54),int(c.height*.42)
    texture=rng.integers(70,200,(h,w),dtype=np.uint8)
    torso[y:y+h,x:x+w]=cv2.cvtColor(cv2.GaussianBlur(texture,(3,3),0),cv2.COLOR_GRAY2BGR)
    mask[y:y+h,x:x+w]=255
    camera=np.column_stack((.6*c.camera_amplitude*np.sin(2*np.pi*c.camera_hz*t+.3),c.camera_amplitude*np.sin(2*np.pi*c.camera_hz*t)))
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),c.fps,(c.width,c.height))
    if not writer.isOpened():
        raise RuntimeError('MP4 writer unavailable.')
    try:
        for i in range(count):
            jump=c.artifact_pixels*np.sin(2*np.pi*1.4*t[i]) if .32*c.duration_seconds<t[i]<.45*c.duration_seconds else 0.
            matrix=np.float32([[1,0,0],[0,1,displacement[i]+jump]])
            moved=cv2.warpAffine(torso,matrix,(c.width,c.height))
            alpha=cv2.warpAffine(mask,matrix,(c.width,c.height)).astype(float)[...,None]/255
            frame=background*(1-alpha)+moved*alpha
            if c.occlusion_fraction and .35*c.duration_seconds<t[i]<.65*c.duration_seconds:
                frame[y:y+h,x:x+int(w*c.occlusion_fraction)]=110
            matrix=np.float32([[1,0,camera[i,0]],[0,1,camera[i,1]]])
            frame=cv2.warpAffine(frame,matrix,(c.width,c.height),borderMode=cv2.BORDER_REFLECT)
            frame+=c.illumination_amplitude*np.sin(2*np.pi*.17*t[i])
            if c.noise_std:
                frame+=rng.normal(0,c.noise_std,frame.shape)
            writer.write(np.clip(frame,0,255).astype(np.uint8))
    finally:
        writer.release()
    metadata={'synthetic':True,'clinical':False,'generator':'challenge-v1','config':asdict(c),
              'ground_truth_bpm':float(instantaneous.mean()),
              'reference_definition':'time-average phase rate including zero-rate pauses; variable-rate clips are not stationary',
              'timestamps_seconds':t.tolist(),'instantaneous_bpm':instantaneous.tolist(),
              'respiratory_displacement_pixels':displacement.tolist(),'camera_translation_pixels':camera.tolist()}
    output_path.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    return metadata

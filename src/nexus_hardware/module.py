'''Nexus Hardware — platform profiles, HAL, capability detection.'''
import os, platform, subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class PlatformProfile:
    name: str; arch: str; cpu_cores: int; ram_mb: int
    has_gpu: bool = False; gpu_name: str = ""
    has_npu: bool = False; sensors: List[str] = field(default_factory=list)
    power_budget_w: float = 10.0; compute_capability: str = ""

PROFILES = {
    "jetson-orin-nano": PlatformProfile("Jetson Orin Nano", "aarch64", 6, 8192,
        has_gpu=True, gpu_name="Orin", power_budget_w=15),
    "jetson-nano": PlatformProfile("Jetson Nano", "aarch64", 4, 4096,
        has_gpu=True, gpu_name="Maxwell", power_budget_w=10),
    "raspberry-pi-5": PlatformProfile("Raspberry Pi 5", "aarch64", 4, 8192, power_budget_w=5),
    "esp32-s3": PlatformProfile("ESP32-S3", "xtensa", 2, 512, sensors=["imu","gpio","adc"]),
    "stm32h7": PlatformProfile("STM32H7", "arm", 1, 1024, sensors=["adc","uart","spi","can"]),
    "jetson-agx-orin": PlatformProfile("Jetson AGX Orin", "aarch64", 12, 32768,
        has_gpu=True, gpu_name="Orin", power_budget_w=60),
    "nvidia-orin-nx": PlatformProfile("NVIDIA Orin NX", "aarch64", 8, 16384,
        has_gpu=True, gpu_name="Orin", power_budget_w=25),
    "beaglebone-black": PlatformProfile("BeagleBone Black", "arm", 1, 512),
    "google-coral": PlatformProfile("Google Coral", "arm", 4, 1024, has_npu=True),
    "intel-ncs2": PlatformProfile("Intel NCS2", "x86_64", 1, 512, has_npu=True),
}

class CapabilityDetector:
    def detect(self) -> PlatformProfile:
        arch = platform.machine(); cpu_count = os.cpu_count() or 1
        try: ram_mb = os.popen("cat /proc/meminfo | grep MemTotal").read().split()[1]
        except: ram_mb = 4096
        if isinstance(ram_mb, str): ram_mb = int(ram_mb) // 1024
        if arch == "aarch64" and cpu_count >= 6: return PROFILES.get("jetson-orin-nano",
            PlatformProfile(f"aarch64-{cpu_count}core", arch, cpu_count, ram_mb))
        if arch == "aarch64": return PROFILES.get("raspberry-pi-5",
            PlatformProfile(f"aarch64-{cpu_count}core", arch, cpu_count, ram_mb))
        return PlatformProfile(f"x86-{cpu_count}core", arch, cpu_count, ram_mb)

class HardwareAbstraction:
    def __init__(self, profile: PlatformProfile):
        self.profile = profile
    def can_run_inference(self, model_params_b: int) -> bool:
        available = self.profile.ram_mb * 1024 * 1024 * 0.3
        return model_params_b * 2 <= available
    def max_batch_size(self, model_params_b: int) -> int:
        available = self.profile.ram_mb * 1024 * 1024 * 0.3
        per_sample = model_params_b * 2
        return max(1, int(available / per_sample))
    def estimate_inference_time(self, params_b: int) -> float:
        gops = params_b * 2
        if self.profile.has_gpu:
            return gops / (self.profile.power_budget_w * 100) * 0.001
        return gops / (self.profile.cpu_cores * 2) * 0.001

def demo():
    print("=== Hardware ===")
    for name, p in list(PROFILES.items())[:5]:
        hal = HardwareAbstraction(p)
        can = hal.can_run_inference(7_000_000_000)
        batch = hal.max_batch_size(7_000_000_000)
        print(f"  {p.name:25s}: {p.ram_mb//1024}GB RAM, GPU={'yes' if p.has_gpu else 'no'}, "
              f"7B inference={'yes' if can else 'no'}, batch={batch}")
    detector = CapabilityDetector()
    detected = detector.detect()
    print(f"\n  Detected: {detected.name} ({detected.arch}, {detected.cpu_cores} cores, "
          f"{detected.ram_mb}MB RAM)")

if __name__ == "__main__": demo()

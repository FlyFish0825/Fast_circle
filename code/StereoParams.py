import numpy as np

class StereoParams:
    """双目相机标定参数（内参 + 畸变 + 立体外参）"""
    
    def __init__(self, K_left, D_left, K_right, D_right, R, T, image_size=(1280, 720)):
        self.image_size = image_size
        self.K_left  = np.asarray(K_left,  dtype=np.float64).reshape(3, 3)
        self.D_left  = np.asarray(D_left,  dtype=np.float64).reshape(-1)
        self.K_right = np.asarray(K_right, dtype=np.float64).reshape(3, 3)
        self.D_right = np.asarray(D_right, dtype=np.float64).reshape(-1)
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(T, dtype=np.float64).reshape(3, 1)
        self.baseline = np.linalg.norm(self.T)

    def __repr__(self):
        fmt = lambda a: np.array2string(a, formatter={'float_kind': lambda x: f"{x:10.4f}"})
        return (
            f"StereoParams(baseline={self.baseline:.2f} mm, size={self.image_size})\n"
            f"左内参:\n{fmt(self.K_left)}\n"
            f"左畸变: {self.D_left}\n"
            f"右内参:\n{fmt(self.K_right)}\n"
            f"右畸变: {self.D_right}\n"
            f"立体旋转:\n{fmt(self.R)}\n"
            f"立体平移: {self.T.ravel()}"
        )


# ==================== 1. Matlab 标定结果（你的原始参数） ====================
image_size = (1280, 720)

K_left_matlab = np.array([
    [723.530136460241, 0.0, 630.921651248589],
    [0.0, 724.210155017952, 332.993256406545],
    [0.0, 0.0, 1.0]
])
D_left_matlab = np.array([0.0693859861215569, -0.0758079441137763, 0.0, 0.0, 0.0])

K_right_matlab = np.array([
    [723.372918090148, 0.0, 612.050169224534],
    [0.0, 723.542234159967, 334.970567694343],
    [0.0, 0.0, 1.0]
])
D_right_matlab = np.array([0.0694875396356727, -0.0680614232359089, 0.0, 0.0, 0.0])

R_matlab = np.array([
    [0.999993023377081, -0.000547017185118819, 0.00369512778717005],
    [0.000539869476935563, 0.999997981999621, 0.00193508078205222],
    [-0.00369617885284314, -0.00193307239501763, 0.999991300708663]
])
T_matlab = np.array([-60.1343628115265, -0.0436072350075528, -0.0799557203620015])

matlab_params = StereoParams(K_left_matlab, D_left_matlab,
                             K_right_matlab, D_right_matlab,
                             R_matlab, T_matlab, image_size)

# ==================== 2. LM 联合优化结果（你的新参数） ====================
K_left_ours = np.array([
    [723.5809, -0.1677, 630.1157],
    [0.0,      724.3112, 332.0110],
    [0.0,      0.0,       1.0]
])
D_left_ours = np.array([0.066197, -0.069312, 0.0, 0.0, 0.0])

K_right_ours = np.array([
    [723.2946, -0.1490, 611.3340],
    [0.0,      723.5472, 334.0096],
    [0.0,      0.0,       1.0]
])
D_right_ours = np.array([0.068497, -0.066832, 0.0, 0.0, 0.0])

R_ours = np.array([
    [1.0000,  0.0006, -0.0038],
    [-0.0006, 1.0000, -0.0019],
    [0.0038,  0.0019,  1.0000]
])
T_ours = np.array([-60.1312, -0.0536, -0.0670])

ours_params = StereoParams(K_left_ours, D_left_ours,
                           K_right_ours, D_right_ours,
                           R_ours, T_ours, image_size)

def main():
    print("========== Matlab 标定结果 ==========")
    print(matlab_params)
    print("\n========== LM 联合优化结果 ==========")
    print(ours_params)

if __name__ == "__main__":
    main()


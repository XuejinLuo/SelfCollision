import numpy as np
import torch
import time
from scipy.optimize import minimize
from utils.ik_solver import IKSolver
from utils.utils import bcolors
from optimize.bounds_calculator import BoundsCalculator

urdf_path = "data/franka_urdf/fr3.urdf"

class OptimizeSolver(object):
    def __init__(self, args, model, test_data_loader):
        self.args = args
        self.num_frames = args.num_frames
        self.robot_operation = args.robot_operation
        self.model = model
        self.model.eval()
        self.is_mamba = args.model.lower() in ["mamba", "smamba"]
        self.test_data_loader = test_data_loader
        self.ik_solver = IKSolver(urdf_path, 100, 1e-4)
        self.distance_weight = args.distance_weight
        self.position_weight = args.position_weight
        self.maxiter = args.maxiter
        self.is_bp = args.model.lower() in ["bp"]
        self.joint_max_position = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32).to("cuda")
        self.joint_min_position = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=torch.float32).to("cuda")
        self.bounds_calculator = BoundsCalculator(cycletime=20)

    def loss_function(self, input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        start_time = time.time()
        output_tensor = self.model.forward(input_tensor)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{bcolors.WARNING}Model Forward pass time: {elapsed_time:.6f} seconds{bcolors.ENDC}")
        print(f"Model Output:\n{output_tensor}\n")
        if self.is_mamba:
            target = torch.full((1, 1, 14), 100.0, dtype=torch.float32).to("cuda")
        else:
            target = torch.full((1, 1), 100.0, dtype=torch.float32).to("cuda")
        distance = torch.norm(output_tensor - target)
        
        if torch.max(output_tensor) > 100:
            zero_grad = np.zeros(7)
            return 0, zero_grad
        
        total_loss = distance * self.distance_weight
        total_loss.backward(retain_graph=True)
        input_grad = input_tensor.grad[0, self.num_frames-1, :].cpu().numpy()
        if self.robot_operation == 1:
            input_grad[:7] = 0
        elif self.robot_operation == 2:
            input_grad[-7:] = 0

        return total_loss.item(), input_grad
    
    def position_error_constraint(self, optimized_last_step, T_result_init):
        input_tensor = torch.tensor(optimized_last_step, dtype=torch.float32, device='cuda')
        robot1_joint_pos = input_tensor[0, self.num_frames - 1, :7] * (self.joint_max_position - self.joint_min_position) + self.joint_min_position
        T_robot1_result = self.ik_solver.compute_fk_tensor(robot1_joint_pos, with_finger=True)
        new_position = T_robot1_result[:3, 3]
        initial_position = T_result_init[:3, 3]
        position_error = torch.norm(new_position - initial_position).item()
        threshold = 1e-1
        return threshold - position_error
    
    def solve(self):
        for batch in self.test_data_loader:
            for i in range(len(batch[0])):  # 遍历每个样本
                first_data = batch[0][i]
                first_labels = batch[1][i]
                first_data = first_data.unsqueeze(0)
                first_labels = first_labels.unsqueeze(0)
                if self.is_bp:
                    first_data = first_data.unsqueeze(1)
                
                last_step_tensor_robot1 = first_data[0, self.num_frames-1, :7]
                last_step_tensor_robot2 = first_data[0, self.num_frames-1, 7:]
                first_data_np = first_data.cpu().numpy()
                manual_input = first_data_np

                def optimize_input():
                    fixed_input = manual_input[0, :self.num_frames-1, :]
                    last_step = manual_input[0, self.num_frames-1, :].flatten()
                    robot1_joint_pos = last_step[:7]
                    robot2_joint_pos = last_step[7:]
                    
                    print(f"{bcolors.OKBLUE}Initial last_step:\n{last_step}{bcolors.ENDC}")
                    robot1_joint_pos_init = last_step_tensor_robot1 * (self.joint_max_position - self.joint_min_position) + self.joint_min_position
                    robot2_joint_pos_init = last_step_tensor_robot2 * (self.joint_max_position - self.joint_min_position) + self.joint_min_position
                    T_robot1_result_init = self.ik_solver.compute_fk_tensor(robot1_joint_pos_init, with_finger=True)
                    T_robot2_result_init = self.ik_solver.compute_fk_tensor(robot2_joint_pos_init, with_finger=True)
                    print(f"{bcolors.OKCYAN}T_robot1_result_init:\n{T_robot1_result_init}{bcolors.ENDC}")
                    print(f"{bcolors.OKGREEN}T_robot2_result_init:\n{T_robot2_result_init}{bcolors.ENDC}")

                    bounds = self.bounds_calculator.calculate_bounds(robot1_joint_pos_init.detach().cpu().numpy(), robot2_joint_pos_init.detach().cpu().numpy())

                    def objective_function(optimized_last_step):
                        optimized_input = np.concatenate([fixed_input, optimized_last_step.reshape(1, 14)], axis=0).reshape(1, self.num_frames, 14)
                        loss, grad = self.loss_function(optimized_input)      
                        return loss, grad.flatten()
                    
                    constraint = {
                        'type': 'ineq',
                        'fun': lambda x: self.position_error_constraint(x, fixed_input, T_robot1_result_init, last_step)
                    }
                    
                    result = minimize(
                        objective_function, 
                        last_step, 
                        method='L-BFGS-B', 
                        jac=True,
                        bounds=bounds,
                        constraints=constraint,
                        options={
                            'maxiter': self.maxiter,
                            'ftol': 1e-6,  # 控制目标函数值的相对变化容忍度
                            'gtol': 1e-6,  # 控制梯度的最大分量的容忍度
                            'maxls': 5    # 每次迭代的最大线搜索步数
                        }
                    )
                    optimized_last_step = result.x
                    return optimized_last_step

                start_time = time.time()
                optimized_last_step = optimize_input()
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"{bcolors.WARNING}Optimize pass time: {elapsed_time:.6f} seconds{bcolors.ENDC}")
                formatted_output = ', '.join(f"{value:.6f}" for value in optimized_last_step)
                print(f"{bcolors.OKBLUE}Optimized last Step Input:\n{formatted_output}{bcolors.ENDC}")

                manual_input[0, self.num_frames-1, :] = optimized_last_step
                manual_input_tensor = torch.tensor(manual_input).to("cuda")
                robot1_joint_pos = manual_input_tensor[0, self.num_frames-1, :7] * (self.joint_max_position - self.joint_min_position) + self.joint_min_position
                robot2_joint_pos = manual_input_tensor[0, self.num_frames-1, 7:] * (self.joint_max_position - self.joint_min_position) + self.joint_min_position
                T_robot1_result = self.ik_solver.compute_fk_tensor(robot1_joint_pos, with_finger=True)
                T_robot2_result = self.ik_solver.compute_fk_tensor(robot2_joint_pos, with_finger=True)
                print(f"{bcolors.OKCYAN}T_robot1_result:\n{T_robot1_result}{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}T_robot2_result:\n{T_robot2_result}{bcolors.ENDC}")

                output = self.model(manual_input_tensor)
                print(f"{bcolors.FAIL}Model Output After Optimization for Sample {i}:{output.detach().cpu().numpy()}\n{bcolors.ENDC}")
                print(f"{bcolors.FAIL}----------------------------------------------------------------{bcolors.ENDC}")


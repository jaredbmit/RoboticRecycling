from enum import Enum
import numpy as np
from pydrake.all import (LeafSystem, JacobianWrtVariable, PointCloud, ImageRgba8U, ImageDepth32F, 
                        Concatenate, AddMultibodyPlantSceneGraph, Role, BaseField, Fields, 
                        RigidTransform, RotationMatrix, AbstractValue, PiecewisePose)

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as Tf
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa_position", 7)
        self.DeclareVectorOutputPort("iiwa_velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV,
            self._G, [0,0,0], self._W, self._W)
        J_G = J_G[:,self.iiwa_start:self.iiwa_end+1] # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)
        
        
class Vision(LeafSystem):
    def __init__(self, station, camera_body_indices, model_path, item_names):
        LeafSystem.__init__(self)
        
        rgb_image = AbstractValue.Make(ImageRgba8U(640,480))
        depth_image = AbstractValue.Make(ImageDepth32F(640,480))
        point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("depth0", depth_image)
        self.DeclareAbstractInputPort("depth1", depth_image)
        self.DeclareAbstractInputPort("rgb0", rgb_image)
        self.DeclareAbstractInputPort("rgb1", rgb_image)
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))
        
        self.DeclareAbstractOutputPort("point_cloud_W", 
            lambda: AbstractValue.Make((0, "", point_cloud)), 
            self.SendSegmentedCloud)
            
        # Crop box for area of interest
        self._crop_lower = np.array([-.5, -.75, .39])
        self._crop_upper = np.array([.5, -.45, .475])
        
        self.item_names = item_names
        self._camera_body_indices = camera_body_indices
        self.cam_info_0 = station.GetSubsystemByName('camera0').depth_camera_info()
        self.cam_info_1 = station.GetSubsystemByName('camera0').depth_camera_info()
            
        ## Load model
        num_classes = len(item_names) + 1
        self.model = self.load_model(num_classes)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
    def load_model(self, num_classes):
        
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, progress=False)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)
        
        return model
        
    def SendSegmentedCloud(self, context, output):
        
        body_poses = self.GetInputPort("body_poses").Eval(context)
        rgb0 = self.GetInputPort("rgb0").Eval(context).data
        rgb1 = self.GetInputPort("rgb1").Eval(context).data
        depth0 = self.GetInputPort("depth0").Eval(context).data
        depth1 = self.GetInputPort("depth1").Eval(context).data
                
        # Put through deep model
        with torch.no_grad():
            predictions = []
            predictions.append(
                self.model([Tf.to_tensor(rgb0[:, :, :3]).to(self.device)]))
            predictions.append(
                self.model([Tf.to_tensor(rgb1[:, :, :3]).to(self.device)]))
        for i in range(2):
            for k in predictions[i][0].keys():
                if k == "masks":
                    predictions[i][0][k] = predictions[i][0][k].mul(
                        255).byte().cpu().numpy()
                else:
                    predictions[i][0][k] = predictions[i][0][k].cpu().numpy()
        
        X_WCs = []
        for idx in self._camera_body_indices:
            X_WCs.append(body_poses[idx])
        
        score, obj_idx, cloud = self.get_merged_masked_pcd(predictions, [rgb0, rgb1], 
           [depth0, depth1], [self.project_depth_to_pC, self.project_depth_to_pC], 
           X_WCs, [self.cam_info_0, self.cam_info_1])
        
        cloud.Crop(self._crop_lower, self._crop_upper)
        
        output.set_value((score, self.item_names[obj_idx], cloud))
        
    def get_merged_masked_pcd(self, predictions, rgb_ims, depth_ims, 
          project_depth_to_pC_funcs, X_WCs, cam_infos, mask_threshold=150):
        """
        predictions: The output of the trained network (one for each camera)
        rgb_ims: RGBA images from each camera
        depth_ims: Depth images from each camera
        project_depth_to_pC_funcs: Functions that perform the pinhole camera operations to convert pixels
            into points. See the analogous function in problem 5.2 to see how to use it.
        X_WCs: Poses of the cameras in the world frame
        """
        # Let's focus on the maximal confidence object
        # Limitation: Assumes object uniqueness
        scores = {}
        for obj_idx in range(len(self.item_names)):
            combined_score = 0
            for p in predictions:
                if obj_idx not in p[0]['labels']:
                    continue
                combined_score += np.max(
                    p[0]['scores'][p[0]['labels'] == obj_idx])
            if combined_score > 0:
                scores[obj_idx] = combined_score
            
        if not bool(scores):
            return 0, -1, PointCloud(0)
        
        obj_idx = max(scores, key=scores.get)
        print("Decided to pick up ", self.item_names[obj_idx])
        
        pcd = []
        for prediction, rgb_im, depth_im, project_depth_to_pC_func, X_WC, cam_info in \
                zip(predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs, cam_infos):

            obj_masks = prediction[0]['masks'][prediction[0]['labels'] == obj_idx]
            mask = obj_masks[0,0]
            idx = np.where(mask >= mask_threshold)
            depth_pts = np.column_stack((idx[0], idx[1], depth_im[idx[0], idx[1]]))
            p_C_obj = project_depth_to_pC_func(depth_pts, cam_info)
            spatial_points = X_WC @ p_C_obj.T
            rgb_points = rgb_im[idx[0], idx[1], 0:3].T

            # You get an unhelpful RunTime error if your arrays are the wrong
            # shape, so we'll check beforehand that they're the correct shapes.
            assert len(spatial_points.shape
                      ) == 2, "Spatial points is the wrong size -- should be 3 x N"
            assert spatial_points.shape[
                0] == 3, "Spatial points is the wrong size -- should be 3 x N"
            assert len(rgb_points.shape
                      ) == 2, "RGB points is the wrong size -- should be 3 x N"
            assert rgb_points.shape[
                0] == 3, "RGB points is the wrong size -- should be 3 x N"
            assert rgb_points.shape[1] == spatial_points.shape[1]

            N = spatial_points.shape[1]
            pcd.append(PointCloud(N, Fields(BaseField.kXYZs | BaseField.kRGBs)))
            pcd[-1].mutable_xyzs()[:] = spatial_points
            pcd[-1].mutable_rgbs()[:] = rgb_points
            # Estimate normals
            pcd[-1].EstimateNormals(radius=0.1, num_closest=30)
            # Flip normals toward camera
            pcd[-1].FlipNormalsTowardPoint(X_WC.translation())

        # Merge point clouds.
        merged_pcd = Concatenate(pcd)
        
        # Get the prediciton score
        avg_score = scores[obj_idx] / len(rgb_ims)

        # Voxelize down-sample.  (Note that the normals still look reasonable)
        return avg_score, obj_idx, merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    
    def project_depth_to_pC(self, depth_pixel, cam_info):
        """
        project depth pixels to points in camera frame
        using pinhole camera model
        Input:
            depth_pixels: numpy array of (nx3) or (3,)
        Output:
            pC: 3D point in camera frame, numpy array of (nx3)
        """
        # switch u,v due to python convention
        v = depth_pixel[:,0]
        u = depth_pixel[:,1]
        Z = depth_pixel[:,2]
        cx = cam_info.center_x()
        cy = cam_info.center_y()
        fx = cam_info.focal_x()
        fy = cam_info.focal_y()
        X = (u-cx) * Z/fx
        Y = (v-cy) * Z/fy
        pC = np.c_[X,Y,Z]
        return pC

    
class GraspSelector(LeafSystem):
    def __init__(self, internal_model, X_WHome):
        LeafSystem.__init__(self)
        
        point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("point_cloud_W", 
            AbstractValue.Make((0, "", point_cloud)))
        
        port = self.DeclareAbstractOutputPort(
            "grasp_selection", 
            lambda: AbstractValue.Make((np.inf, RigidTransform(), "")), 
            self.SelectGrasp)
        port.disable_caching_by_default()
        
        self._internal_model = internal_model
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng()
        self.X_WHome = X_WHome
        
    # Taken from manipulation/clutter.py - slightly modified
    def GraspCandidateCost(self, diagram, context, cloud, wsg_body_index=None, plant_system_name="plant",
                           scene_graph_system_name="scene_graph", adjust_X_G=False):
        plant = diagram.GetSubsystemByName(plant_system_name)
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
        if wsg_body_index:
            wsg = plant.get_body(wsg_body_index)
        else:
            wsg = plant.GetBodyByName("body")
            wsg_body_index = wsg.index()

        X_G = plant.GetFreeBodyPose(plant_context, wsg)

        # Transform cloud into gripper frame
        X_GW = X_G.inverse()
        p_GC = X_GW @ cloud.xyzs()

        # Crop to a region inside of the finger box.
        crop_min = [-.05, 0.1, -0.00625]
        crop_max = [.05, 0.1125, 0.00625]
        indices = np.all((crop_min[0] <= p_GC[0, :], p_GC[0, :] <= crop_max[0],
                          crop_min[1] <= p_GC[1, :], p_GC[1, :] <= crop_max[1],
                          crop_min[2] <= p_GC[2, :], p_GC[2, :] <= crop_max[2]),
                         axis=0)

        if adjust_X_G and np.sum(indices) > 0:
            p_GC_x = p_GC[0, indices]
            p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
            X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
            plant.SetFreeBodyPose(plant_context, wsg, X_G)
            X_GW = X_G.inverse()

        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

        # Check collisions between the gripper and the sink
        if query_object.HasCollisions():
            cost = np.inf
            return cost

        # Check collisions between the gripper and the point cloud. `margin`` must
        # be smaller than the margin used in the point cloud preprocessing.
        margin = 0.0
        for i in range(cloud.size()):
            distances = query_object.ComputeSignedDistanceToPoint(cloud.xyz(i),
                                                                  threshold=margin)
            if distances:
                cost = np.inf
                return cost

        n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])

        # Reward sum |dot product of normals with gripper x|^2
        cost = -np.sum(n_GC[0, :]**2)
        return cost
        
    def GenerateAntipodalGraspCandidate(self, diagram, context, cloud, rng, wsg_body_index=None,
                                    plant_system_name="plant", scene_graph_system_name="scene_graph"):
        """
        Picks a random point in the cloud, and aligns the robot finger with the
        x/y projection of the normal of that pixel. Perturbs z a little to find better grasps.
        """
        plant = diagram.GetSubsystemByName(plant_system_name)
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
        if wsg_body_index:
            wsg = plant.get_body(wsg_body_index)
        else:
            wsg = plant.GetBodyByName("body")
            wsg_body_index = wsg.index()

        if cloud.size() < 1:
            return np.inf, None

        index = rng.integers(0, cloud.size() - 1)

        # Use S for sample point/frame.
        p_WS = cloud.xyz(index)
        n_WS = cloud.normal(index)

        if not np.isclose(np.linalg.norm(n_WS), 1.0):
            print("Skipping: ", f"Normal has magnitude: {np.linalg.norm(n_WS)}")
            return np.inf, None

        # Modification: always keep gripper y aligned with world -z
        Gx = np.array([n_WS[0], n_WS[1], 0]) # Project onto XY plane
        Gx = Gx / np.linalg.norm(Gx)
        Gy = np.array([0.0, 0.0, -1.0]) # World downward
        Gz = np.cross(Gx, Gy)
        R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
        p_GS_G = [0.054 - 0.01, 0.1, 0] # Position of sample end wrt gripper
        p_WG = p_WS - R_WG @ p_GS_G

        # Try vertical perturbations if necessary
        for z in [0, -.01, .01, -.02, .02]:
            p_WG_2 = p_WG + np.array([0, 0, z])
            X_G = RigidTransform(R_WG, p_WG_2)
            plant.SetFreeBodyPose(plant_context, wsg, X_G)
            cost = self.GraspCandidateCost(diagram, context, cloud, adjust_X_G=True)
            X_G = plant.GetFreeBodyPose(plant_context, wsg)
            if np.isfinite(cost):
                return cost, X_G

        return np.inf, None
        
    def SelectGrasp(self, context, output):
        
        score, item_selection, down_sampled_pcd = self.GetInputPort("point_cloud_W").Eval(context)
        if score < 0.75:
            output.set_value((np.inf, self.X_WHome, item_selection))
            return

        costs = []
        X_Gs = []
        for i in range(25):
            cost, X_G = self.GenerateAntipodalGraspCandidate(
                self._internal_model, self._internal_model_context,
                down_sampled_pcd, self._rng)
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            output.set_value((np.inf, self.X_WHome))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best], item_selection))
         
        
class GarbageType(Enum):
    TRASH = 0
    RECYCLE = 1
    ORGANIC = 2
            
            
class PlannerState(Enum):
    DEBUG = -1
    WAIT = 0
    SELECT_GRASP = 1
    PICK = 2
    PLACE = 3
    HOME = 4
    CLOSE = 5
    OPEN = 6
            
            
class Planner(LeafSystem):
    def __init__(self, plant, Xs, garbage_map):
        LeafSystem.__init__(self)
        
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
        self._grasp_index = self.DeclareAbstractInputPort("grasp_selection", 
            AbstractValue.Make((np.inf, RigidTransform(), ""))).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()
        self.DeclareVectorInputPort("iiwa_position", 7)
        
        self.X_WHome = Xs['Home']
        self.X_WRecycle = Xs[GarbageType.RECYCLE]
        self.X_WTrash = Xs[GarbageType.TRASH]
        self.X_WOrganic = Xs[GarbageType.ORGANIC]
        
        self.eps = 1e-4
        self.init = True
        self.prev_time = None
        self.mode = PlannerState.WAIT
        self.wsg_des = np.array([0.107])
        self.traj_WG = PiecewisePose.MakeLinear([0, np.inf], [RigidTransform(), RigidTransform()])
        self.garbage_type = GarbageType.RECYCLE
        self.garbage_map = garbage_map
        self.num_successful_picks = 0
        
        self.DeclareVectorOutputPort("wsg_position", 1,
            lambda context, output: output.SetFromVector([self.wsg_des]))
        self.DeclareVectorOutputPort("V_WG", 6, self.SendTrajV)
        
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)
        
    def Update(self, context, state):
        if self.mode == PlannerState.DEBUG:
            return
        
        if self.init:
            self.prev_time = context.get_time()
            self.init = False
            return
        
        if self.num_successful_picks == 4:
            print("Finished")
            return
        
        if self.mode == PlannerState.WAIT:
            if context.get_time() - self.prev_time > 2.:
                self.mode = PlannerState.SELECT_GRASP
            
        if self.mode == PlannerState.SELECT_GRASP:
            print("Selecting grasp")
            [cost, X_WD, item_selection] = self.get_input_port(self._grasp_index).Eval(context)
            self.garbage_type = self.garbage_map[item_selection]
            if cost == np.inf:
                self.mode = PlannerState.WAIT
                self.prev_time = context.get_time()
            else:
                self.traj_WG = self.MakePickingTraj(context, X_WD)
                self.mode = PlannerState.PICK
                
        if self.mode == PlannerState.PICK:
            if self.TrajDone(context):
                self.mode = PlannerState.CLOSE
                self.wsg_des = np.array([0.0])
            
        # Need a better way to determine when it's closed.
        if self.mode == PlannerState.CLOSE:
            if context.get_time() - self.prev_time > 2.:
                self.traj_WG = self.MakePlacingTraj(context, self.garbage_type)
                self.mode = PlannerState.PLACE
            
        if self.mode == PlannerState.PLACE:
            if self.TrajDone(context):
                self.mode = PlannerState.OPEN
                self.wsg_des = np.array([0.107])
            
        if self.mode == PlannerState.OPEN:
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
            if wsg_state[0] > 0.09:
                self.traj_WG = self.MakeHomeTraj(context)
                self.mode = PlannerState.HOME
                
        if self.mode == PlannerState.HOME:
            if self.TrajDone(context):
                self.mode = PlannerState.SELECT_GRASP
                self.num_successful_picks += 1
            

    def TrajDone(self, context):
        """
        Checks if we have completed our desired trajectory
            - Time elapsed
            - We're at the desired end pose
        If time is up and we're no longer moving, but we haven't 
            gone to our desired location, then make a micro-adjustment.
        """
        time = context.get_time()
        X_WG = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        X_WEnd = self.traj_WG.GetPose(np.inf)
        if X_WG.IsNearlyEqualTo(X_WEnd, self.eps):
            return True
        elif self.traj_WG.end_time() < time:
            self.traj_WG = PiecewisePose.MakeLinear([time, time+0.2], [X_WG, X_WEnd])
        
        return False

    def MakePickingTraj(self, context, X_WD):
        """
        Creates a basic picking trajectory
        """
        time = context.get_time()
        X_WG = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        X_WPrepick = RigidTransform(RotationMatrix(), [0, 0, 0.1]) @ X_WD
        traj_WG = PiecewisePose.MakeLinear([time, time+1, time+1.5], [X_WG, X_WPrepick, X_WD])
        
        return traj_WG
    
    def MakePlacingTraj(self, context, garbage_type):
        """
        Creates a placing trajectory to either trash, recycle, or organic bin
        """
        time = context.get_time()
        X_WG = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        X_WPostpick = RigidTransform(RotationMatrix(), [0, 0, 0.1]) @ X_WG
        if garbage_type == GarbageType.TRASH:
            traj_WG = PiecewisePose.MakeLinear([time, time+0.5, time+1.5, time+4], 
                          [X_WG, X_WPostpick, self.X_WHome, self.X_WTrash])
        elif garbage_type == GarbageType.RECYCLE:
            traj_WG = PiecewisePose.MakeLinear([time, time+0.5, time+1.5, time+4], 
                          [X_WG, X_WPostpick, self.X_WHome, self.X_WRecycle])
        elif garbage_type == GarbageType.ORGANIC:
            traj_WG = PiecewisePose.MakeLinear([time, time+0.5, time+1.5, time+4], 
                          [X_WG, X_WPostpick, self.X_WHome, self.X_WOrganic])
            
        return traj_WG
    
    def MakeHomeTraj(self, context):
        """
        Creates a basic trajectory for going to home position
        """
        time = context.get_time()
        X_WG = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        traj_WG = PiecewisePose.MakeLinear([time, time+2.5], [X_WG, self.X_WHome])
        
        return traj_WG

    def SendTrajV(self, context, output):
        """
        Callback for evaluating V_WG. Makes derivative of trajectory and publishes
        the V_WG corresponding to current time.
        """
        time = context.get_time()
        V_WG = self.traj_WG.GetVelocity(time)
        output.SetFromVector(V_WG)
        
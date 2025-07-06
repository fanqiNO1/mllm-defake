from swift.plugin import ORM

from mllm_defake.finetune.rewards.utils import cal_iou, check_format, check_label

class BaseReward(ORM):
    
    # reward base, can be used to shift the reward distribution
    reward_base = 0
    # iout reward
    iou_reward_weight = 1.0
    iou_reward_exponent = 1.0
    # label reward
    label_correct_reward_weight = 1.0
    label_incorrect_reward_weight = -1.0
    # format reward
    format_correct_reward_weight = 1.0
    format_incorrect_reward_weight = -1.0

    def __call__(self, completions: list[str], objects: list[dict], **kwargs):
        """Calculate the reward for the given completions and objects.
        
        Args:
            completions (list[str]): List of completions to evaluate.
            objects (list[dict]): List of objects to evaluate against. There are two keys in each dict:
                - "ref": list[str], the reference text for the object.
                - "bbox": list[list[float]], the bounding box coordinates for the object.
            The `objects` is the same as the data format used in SFT stage.
        """
        rewards = []
        for this_completion, this_objects in zip(completions, objects):
            reward = self.reward_base
            # calculate iou reward
            # if this_objects is empty, iou is 0
            iou = cal_iou(this_completion, this_objects)
            iou_reward = self.iou_reward_weight * (iou ** self.iou_reward_exponent)
            reward += iou_reward
            # calculate label reward
            # if this_objects is not empty, actual label is `ai generated`
            is_label_correct = check_label(this_completion, this_objects)
            if is_label_correct:
                label_reward = self.label_correct_reward_weight
            else:
                label_reward = self.label_incorrect_reward_weight
            reward += label_reward
            # calculate format reward
            is_format_correct = check_format(this_completion)
            if is_format_correct:
                format_reward = self.format_correct_reward_weight
            else:
                format_reward = self.format_incorrect_reward_weight
            reward += format_reward
            rewards.append(reward)
        return rewards

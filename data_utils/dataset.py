from torch.utils import data
from tqdm import tqdm
import codecs as cs
import numpy as np
import os
from os.path import join as pjoin
import json
import torch.nn.functional as F
import spacy

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, word_vectorizer):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.motion_texts = []
        self.data_masks = []
        self.lengths = []
        id_list = []
        self.loaded_ids = []
        self.word_vectorizer = word_vectorizer
        self.nlp = spacy.load("en_core_web_sm")
        self.word_embeddings = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        print('id list', len(id_list))
        small_motions = 0
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                loaded_texts = self._load_texts(name)
                loaded_embeddings = []
                for sntnc in loaded_texts:
                    word_list, pos_list = self._process_text(sntnc)
                    tokens = [
                        "%s/%s" % (word_list[i], pos_list[i])
                        for i in range(len(word_list))
                    ]
                    word_embeddings, pos_one_hots, sent_len, tokens = self._vectorize_tokens(tokens)
                    loaded_embeddings.append({
                        'word_embeddings': word_embeddings,
                        'pos_one_hots': pos_one_hots,
                        'sent_len': sent_len,
                        'tokens': tokens
                    })
                self.motion_texts.extend(loaded_texts)
                
                num_texts = len(loaded_texts)
                if motion.shape[0] < opt.max_motion_length:
                    #print('motion shape: ', motion.shape[0], opt.max_motion_length)
                    orig_len = motion.shape[0]
                    pad_amt = opt.max_motion_length - orig_len
                    motion = np.pad(motion, ((0, pad_amt), (0, 0)), mode = 'constant', constant_values = 0)
                    small_motions += 1
                    mask = np.zeros(opt.max_motion_length).astype(float)
                    mask[:orig_len] = 1.0
                else:
                    orig_len = motion.shape[0]
                    motion = motion[:opt.max_motion_length, :]
                    mask = np.zeros(opt.max_motion_length).astype(float)
                    mask[:orig_len] = 1.0
                loaded_embeddings = [emb | {'original_motion_length': orig_len} for emb in loaded_embeddings]
                self.data.extend([motion] * num_texts)
                self.word_embeddings.extend(loaded_embeddings)
                self.data_masks.extend([mask] * num_texts)
                self.loaded_ids.extend([name] * num_texts)
            except Exception as e:
                print('Dataset load exception: ', e)
                # Some motion may not exist in KIT dataset
                pass

        #self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print(f'Motion shape (B, T, D): ({len(self.data)}, {self.data[0].shape[0]}, {self.data[0].shape[1]})')
        #print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))
        print("Total number of motions {}".format(len(self.data)))
        print("Total number of small motions: {}".format(small_motions))

    def _load_texts(self, name):
        text_path = pjoin(self.opt.text_dir, name + '.txt')
        texts = []
        if os.path.exists(text_path):
            with cs.open(text_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('#')
                    caption = parts[0].strip()
                    if caption:
                        texts.append(caption)
        else:
            ValueError(f'Folder {self.opt.text_dir} not found under dataset folder')
        return texts
    
    def _process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list
    
    def _vectorize_tokens(self, tokens, max_text_len: int = 20):
        # wrap with sos/eos and pad/crop, exactly like MLD
        if len(tokens) < max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)                     # true length before padding
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)                     # == max_text_len + 2

        # build word_embeddings and pos_one_hots for model evaluation
        word_emb_list = []
        pos_oh_list = []
        for tok in tokens:
            word_emb, pos_oh = self.word_vectorizer[tok]
            word_emb_list.append(word_emb[None, :])    # [1, word_dim]
            pos_oh_list.append(pos_oh[None, :])        # [1, pos_dim]

        word_embeddings = np.concatenate(word_emb_list, axis=0).astype(np.float32)  # [L, word_dim]
        pos_one_hots   = np.concatenate(pos_oh_list, axis=0).astype(np.float32)     # [L, pos_dim]

        return word_embeddings, pos_one_hots, sent_len, tokens

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def __len__(self):
        #return self.cumsum[-1]
        return len(self.data)
    
    def __getitem__(self, item):
        motion_id = item
        motion = self.data[motion_id]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        motion_file_id = self.loaded_ids[motion_id]
        #texts = self._load_texts(motion_file_id)
        text = self.motion_texts[item]
        #word_embeddings, pos_one_hots, sent_len, tokens = self._vectorize_tokens(tokens)
        #text = texts[0] if len(texts) > 0 else ""
        motion_mask = self.data_masks[motion_id]
        word_embeddings = self.word_embeddings[item]

        return {
            'motion': motion,
            'motion_mask': motion_mask,
            'file_id': motion_file_id,
            'text': text,
            'word_embeddings': word_embeddings['word_embeddings'],
            'pos_one_hots': word_embeddings['pos_one_hots'],
            'sent_len': word_embeddings['sent_len'],
            'tokens': word_embeddings['tokens'],
            'original_motion_length': word_embeddings['original_motion_length']
        }
    


    def __getitemsnippet__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        assert motion.shape[0] == self.opt.window_size and motion.shape[1] == 263, f"Bad T at idx {idx}: {motion.shape}"
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        motion_file_id = self.loaded_ids[motion_id]
        #texts = self._load_texts(motion_file_id)
        text = self.motion_texts[item]
        #text = texts[0] if len(texts) > 0 else ""

        return {
            'motion': motion,
            'file_id': motion_file_id,
            'text': text
        }
    

class PartMotionDataset(MotionDataset):
    def __init__(self, opt, mean, std, split_file, word_vectorizer):
        super().__init__(opt, mean, std, split_file, word_vectorizer)
        self.joints_num = opt.joints_num
        assert self.joints_num == 22, "This version assumes HumanML3D with 22 joints."

        self.part_groups = {
            "root": [],
            "torso": [3, 6, 9, 12, 15],
            "left_arm": [13, 16, 18, 20],
            "right_arm": [14, 17, 19, 21],
            "left_leg": [1, 4, 7, 10],
            "right_leg": [2, 5, 8, 11],
        }
        self.part_names = list(self.part_groups.keys())
        self.part_feature_indices = self._build_part_feature_indices()
        self.d_part_max = max(len(v) for v in self.part_feature_indices.values())

        dataset_mappings = {
            "part_names": self.part_names,
            "part_feature_indices": {
                k: v.tolist() for k, v in self.part_feature_indices.items()
            },
            "d_part_max": self.d_part_max,
            "joints_num": self.joints_num
        }

        with open(pjoin(self.opt.meta_dir, "part_mapping.json"), "w") as fp:
            json.dump(dataset_mappings, fp, indent = 4)

    def _build_part_feature_indices(self):
        J = self.joints_num

        ric_start = 4
        ric_end = ric_start + (J - 1) * 3

        rot_start = ric_end
        rot_end = rot_start + (J - 1) * 6

        vel_start = rot_end
        vel_end = vel_start + J * 3

        foot_start = vel_end

        part_indices = {}

        for part_name, joints in self.part_groups.items():
            idxs = []

            if part_name == "root":
                idxs.extend(range(0, 4))  # root_rot_velocity, root_linear_velocity, root_y

            for j in joints:
                if j > 0:
                    ric_offset = ric_start + (j - 1) * 3
                    idxs.extend(range(ric_offset, ric_offset + 3))

                    rot_offset = rot_start + (j - 1) * 6
                    idxs.extend(range(rot_offset, rot_offset + 6))

                vel_offset = vel_start + j * 3
                idxs.extend(range(vel_offset, vel_offset + 3))

            if part_name == "left_leg":
                idxs.extend([foot_start, foot_start + 1])
            elif part_name == "right_leg":
                idxs.extend([foot_start + 2, foot_start + 3])

            part_indices[part_name] = np.array(sorted(idxs), dtype=np.int64)

        return part_indices

    def __getitem__(self, item):
        motion_data = super().__getitem__(item)   # (T, 263), already normalized
        motion = motion_data['motion']
        #assert motion.shape[0] == self.opt.window_size and motion.shape[1] == 263, f"Bad T at idx {item}: {motion.shape}"
        assert motion.shape[0] == self.opt.max_motion_length and motion.shape[1] == 263, f"Bad T at idx {item}: {motion.shape}"

        T, D = motion.shape
        P = len(self.part_names)
        motion_parts = np.zeros((T, P, self.d_part_max), dtype=np.float32)

        for p, part_name in enumerate(self.part_names):
            idxs = self.part_feature_indices[part_name]
            part_feat = motion[:, idxs]
            motion_parts[:, p, :part_feat.shape[1]] = part_feat
        

        return {
            "motion": motion.astype(np.float32),              # (T, D) = (T, 263)
            "motion_parts": motion_parts.astype(np.float32),  # (T, P, D_part_max)
            #"texts": motion_data['texts'],
            'file_id': motion_data['file_id'],
            'text': motion_data['text'],
            'motion_mask': motion_data['motion_mask'],
            'word_embeddings': motion_data['word_embeddings'],
            'pos_one_hots': motion_data['pos_one_hots'],
            'sent_len': motion_data['sent_len'],
            'tokens': motion_data['tokens'],
            'original_motion_length': motion_data['original_motion_length']
        }
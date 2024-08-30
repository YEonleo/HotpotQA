import torch
import torch.nn.functional as F
import numpy as np
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

def jsd(p, q, base=torch.tensor(2.0)):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p, m, reduction='batchmean', log_target=True) + F.kl_div(q, m, reduction='batchmean', log_target=True)) / torch.log(base)

class Baseline:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=24):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.stop_word_ids = []
        self.stopping_criteria = None

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        def configure_cuda_settings():
            if self.num_gpus == "auto":
                return {"device_map": "auto"}
            self.num_gpus = int(self.num_gpus)
            if self.num_gpus > 1:
                return {
                    "device_map": "auto",
                    "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)}
                }
            return {}

        def validate_device():
            if self.device not in ["cuda", "cpu"]:
                raise ValueError(f"Invalid device: {self.device}")

        # Validate device first
        validate_device()

        # Common settings for all devices
        kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }

        # Configure CUDA-specific settings
        if self.device == "cuda":
            kwargs.update(configure_cuda_settings())

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        # Move model to GPU if single GPU is used
        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            self.stop_word_ids.extend(stop_word_ids)
            print(f"Added stop word: {stop_word} with the ids {stop_word_ids}", flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria([self.tokenizer.encode('\n' + word)[3:] for word in stop_words]))

    def generate(self, input_text, mode, layer, all_choices = None, max_new_tokens=64):
        with torch.no_grad():
            prompt_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

            initial_length_prompt = prompt_ids.shape[1]


            if mode == "Baseline":
                return self._model_generate_base(prompt_ids, initial_length_prompt, max_new_tokens)
            elif mode == "Baseline_for_logit_argMax":
                return self._model_generate_base_argmax(prompt_ids, initial_length_prompt,all_choices, max_new_tokens, layer)
            elif mode == "DOLA":
                return self._model_generate_dola(prompt_ids, initial_length_prompt,all_choices, max_new_tokens)
            elif mode == "DOLA_jsd":
                return self._model_generate_dola_jsd(prompt_ids, initial_length_prompt,all_choices, max_new_tokens)
            
            return self._generate(prompt_ids)

    def _model_generate_base_argmax(self, prompt_ids, initial_length_prompt, all_choices, max_new_tokens, layer=None):
        outputs = self.model(prompt_ids, output_hidden_states=True)
        
        if layer is None:
            logits = outputs.logits[:, -1, :].squeeze(0)  # 최종 출력 레이어의 logits 사용
        else:
            hidden_state = outputs.hidden_states[layer]  # 지정된 레이어의 히든 스테이트 사용
            logits = self.model.lm_head(hidden_state[:, -1, :])  # lm_head를 사용하여 logits로 변환

        # 선택지의 토큰 ID 가져오기
        choice_ids = [self.tokenizer.convert_tokens_to_ids(choice) for choice in all_choices]

        # 선택지에 해당하는 logits만 추출
        logits_for_choices = logits[0, choice_ids]

        # 소프트맥스 적용
        probs = (
            torch.nn.functional.softmax(logits_for_choices, dim=0)
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

        return np.argmax(probs)
    
    def _model_generate_dola(self, prompt_ids, initial_length_prompt, all_choices, max_new_tokens, layer=None):
        outputs = self.model(prompt_ids, output_hidden_states=True)
        
        # Final hidden state logits
        final_hidden_state = self.model.lm_head(outputs.hidden_states[-1][:, -1, :])
        choice_ids = [self.tokenizer.convert_tokens_to_ids(choice) for choice in all_choices]
        logits_for_final_layer = final_hidden_state[0, choice_ids]
        final_probs = torch.softmax(logits_for_final_layer, dim=0).detach().cpu().numpy()
        
        kl_divergences = []
        layer_probs = []
        
        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            if layer_idx >= 1:
                logits = self.model.lm_head(hidden_state[:, -1, :])
                logits_for_choices = logits[0, choice_ids]
                probs = torch.softmax(logits_for_choices, dim=0).detach().cpu().numpy()
                
                # KL divergence between the final layer probabilities and current layer
                kl_div = F.kl_div(
                    torch.log_softmax(logits_for_choices, dim=0), 
                    torch.softmax(logits_for_final_layer, dim=0),
                    reduction='batchmean'
                )
                kl_divergences.append((kl_div.item(), layer_idx))
                layer_probs.append(probs)
        
        # Find the layer with the maximum KL divergence
        max_kl_div_value, max_kl_div_index = max(kl_divergences, key=lambda x: x[0])
        max_kl_div_layer_prob = layer_probs[max_kl_div_index - 1]
        
        # Idea: comparing final_probs with the most divergent layer's probs
        idea_probs = final_probs - max_kl_div_layer_prob

        return np.argmax(idea_probs)
    
    def _model_generate_dola_jsd(self, prompt_ids, initial_length_prompt, all_choices, max_new_tokens, layer=None):
        outputs = self.model(prompt_ids, output_hidden_states=True)
        
        # Final hidden state logits
        final_hidden_state = self.model.lm_head(outputs.hidden_states[-1][:, -1, :])
        choice_ids = [self.tokenizer.convert_tokens_to_ids(choice) for choice in all_choices]
        logits_for_final_layer = final_hidden_state[0, choice_ids]
        final_probs = torch.softmax(logits_for_final_layer, dim=0).detach().cpu().numpy()
        
        js_divergences = []
        layer_probs = []
        
        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            if layer_idx >= 1:
                logits = self.model.lm_head(hidden_state[:, -1, :])
                logits_for_choices = logits[0, choice_ids]
                probs = torch.softmax(logits_for_choices, dim=0).detach().cpu().numpy()
                
                # JSD between the final layer probabilities and current layer
                jsd_value = jsd(
                    torch.softmax(torch.tensor(logits_for_choices), dim=0),
                    torch.softmax(torch.tensor(logits_for_final_layer), dim=0)
                )
                js_divergences.append((jsd_value.item(), layer_idx))
                layer_probs.append(probs)
        
        # Find the layer with the maximum JSD value
        max_jsd_value, max_jsd_index = max(js_divergences, key=lambda x: x[0])
        max_jsd_layer_prob = layer_probs[max_jsd_index - 1]
        
        # Idea: comparing final_probs with the most divergent layer's probs
        idea_probs = final_probs - max_jsd_layer_prob

        return np.argmax(idea_probs)

    def _model_generate_base(self, prompt_ids, initial_length_prompt, max_new_tokens):
        outputs = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            #return_dict_in_generate=True
        )
        return self.tokenizer.decode(outputs[0,initial_length_prompt:], skip_special_tokens=True)



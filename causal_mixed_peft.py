from peft.mixed_model import PeftMixedModel
import torch
from peft import PeftConfig, PeftType
from typing import Any, Literal, Optional, Union
from contextlib import contextmanager, nullcontext
import os
from peft.utils import get_peft_model_state_dict, SAFETENSORS_WEIGHTS_NAME, id_tensor_storage
import collections
from safetensors.torch import save_file as safe_save_file

class RecPeftMixedModelForCausalLM(PeftMixedModel):


    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.special_peft_forward_args = {"adapter_names"}
        
    def get_base_model(self):
        return self.base_model.model

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If the base model has a method called _enable_peft_forward_hooks, it is invoked as a context. Otherwise, this
        # runs without any changes
        if hasattr(self.base_model, "_enable_peft_forward_hooks"):
            with self.base_model._enable_peft_forward_hooks(*args, **kwargs):
                yield
            return
        else:
            # nothing to enable
            yield
            return

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
                If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
                embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
                and automatically sets the boolean flag. This only works for 🤗 transformers models.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            path_initial_model_for_weight_conversion (`str, *optional*`):
                The path to the initialized adapter, which is obtained after initializing the model with
                PiSSA/CorDA/OLoRA and before performing any training. When `path_initial_model_for_weight_conversion`
                is not None, the difference in adapter before and after fine-tuning is calculated. This difference can
                be represented as the parameters of a standard LoRA adapter. Using this converted adapter does not
                require changes to the base model, thus conveniently allowing the use of multiple PiSSA/CorDA/OLoRA
                adapters with LoRA adapters, and the activation or deactivation of any adapters. Note that this
                conversion is not supported if `rslora` is used in combination with `rank_pattern` or `alpha_pattern`.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.

        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        def save_mutated_as_lora(peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs):
            if peft_config.use_rslora and (peft_config.rank_pattern or peft_config.alpha_pattern):
                msg = (
                    "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                    "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
                )
                raise ValueError(msg)

            if not any(
                str(peft_config.init_lora_weights).lower().startswith(prefix)
                for prefix in ["pissa", "corda", "olora", "true"]
            ):
                warnings.warn(
                    "`path_initial_model_for_weight_conversion` only works for converting a PiSSA/CorDA/OLoRA adapter to "
                    "a LoRA adapter"
                )
            initial_adapter_name = os.path.basename(path_initial_model_for_weight_conversion)
            try:
                self.load_adapter(
                    os.path.dirname(path_initial_model_for_weight_conversion),
                    subfolder=initial_adapter_name,
                    adapter_name=initial_adapter_name,
                )
                is_pissa = str(self.peft_config[initial_adapter_name].init_lora_weights).lower().startswith("pissa")
                is_corda = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "corda"
                is_olora = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "olora"
                if is_pissa or is_corda or is_olora:
                    raise ValueError(
                        "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                        "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                        "residual model."
                    )
                output_state_dict = self.base_model.subtract_mutated_init(
                    output_state_dict, initial_adapter_name, kwargs
                )
            finally:
                self.delete_adapter(initial_adapter_name)
            return output_state_dict

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            #self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
                # Safetensors does not allow tensor aliasing.
                # We're going to remove aliases before saving
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    # Sometimes in the state_dict we have non-tensor objects.
                    # e.g. in bitsandbytes we have some `str` objects in the state_dict
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        # In the non-tensor case, fall back to the pointer of the object itself
                        ptrs[id(tensor)].append(name)

                # These are all the pointers of shared tensors.
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                for _, names in shared_ptrs.items():
                    # Here we just clone the shared tensors to avoid tensor aliasing which is
                    # not supported in safetensors.
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    if not peft_config.use_rslora:
                        peft_config.lora_alpha *= 2
                    else:
                        # with rslora, we have scaling = alpha / sqrt(r), we thus adjust alpha to keep the same scaling
                        peft_config.lora_alpha *= 2**0.5

                    if peft_config.rank_pattern:
                        peft_config.rank_pattern = {key: 2 * val for key, val in peft_config.rank_pattern.items()}
                    if peft_config.alpha_pattern:
                        peft_config.alpha_pattern = {key: 2 * val for key, val in peft_config.alpha_pattern.items()}

                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.peft_config['default']
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            # some archs require max_cache_len to re-initialize the cache
            if input_ids is not None:
                max_cache_len = input_ids.shape[1] + peft_config.num_virtual_tokens
            else:
                max_cache_len = inputs_embeds.shape[1] + peft_config.num_virtual_tokens
            kwargs["past_key_values"] = self.get_prompt(batch_size, max_cache_len=max_cache_len)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        elif peft_config.peft_type == PeftType.CPT:
            return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _cpt_forward(self, input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs):
        # Extract labels from kwargs
        labels = kwargs.pop("labels")
        device = [i.device for i in [input_ids, inputs_embeds, labels] if i is not None][0]
        # Extract input_type_mask from kwargs and move it to the same device as labels
        if "input_type_mask" in kwargs.keys():
            input_type_mask = kwargs.pop("input_type_mask").to(device)
        else:
            if input_ids is None:
                N_tokens = inputs_embeds.shape[1]
            else:
                N_tokens = input_ids.shape[1]
            input_type_mask = torch.ones((batch_size, N_tokens)).to(device) * 4

        cpt_token_ids = peft_config.cpt_token_ids
        cpt_tokens_type_mask = peft_config.cpt_tokens_type_mask

        # Generate embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # Get prompt and concatenate with input embeddings
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        # If labels are provided, generate prefix labels and type mask
        cpt_labels = None
        if labels is not None:
            # Generate prefix labels and concatenate with the input labels
            prefix_labels = torch.Tensor(cpt_token_ids).long().view(1, -1)
            prefix_labels = prefix_labels.repeat(batch_size, 1).to(labels.device)
            cpt_labels = torch.cat((prefix_labels, labels), dim=1)
            # Generate prefix type mask and shift input type mask values to avoid conflicts
            prefix_type_mask = torch.Tensor(cpt_tokens_type_mask).long().view(1, -1)
            prefix_type_mask = prefix_type_mask.repeat(batch_size, 1).to(labels.device)
            adjusted_input_type_mask = input_type_mask
            adjusted_input_type_mask[adjusted_input_type_mask > 0] += prefix_type_mask.max()
            # Concatenate prefix and shifted input type masks
            cpt_type_mask = torch.cat((prefix_type_mask, adjusted_input_type_mask), dim=1)
            # Identify valid label positions and mask invalid ones with -100
            labels_idx = (cpt_type_mask > 0) & (cpt_type_mask % 4 == 0)
            cpt_labels[~labels_idx] = -100
            # Update kwargs with the modified labels

        kwargs["labels"] = cpt_labels
        # Pass the modified inputs to the base model
        base_model_output = self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        if labels is None:
            return base_model_output
        else:
            # Calculate the loss using the custom CPT loss function
            cpt_embedding = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
            base_model_output = cpt_embedding.calculate_loss(
                base_model_output, cpt_labels, cpt_type_mask, self.peft_config["default"]
            )
            return base_model_output

    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(*args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(*args, **kwargs)
            else:
                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        if packaging.version.parse(transformers.__version__) > packaging.version.parse("4.43.3"):
            # https://github.com/huggingface/transformers/pull/31445
            transformers_new_cache_archs.append("bloom")

        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        # heuristic to determine if we're in 'prefill stage' (when the KV cache is filled with the values from the
        # initial input)
        is_prefill = (model_kwargs.get("cache_position") is not None) and (model_kwargs["cache_position"][0] == 0)

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs.get("past_key_values", None) is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if (attention_mask := model_kwargs.get("attention_mask", None)) is not None:
                if isinstance(attention_mask, dict):
                    # see: https://github.com/huggingface/transformers/pull/37866
                    # For now, just deal with the case of a single attention mask
                    if len(attention_mask) != 1:
                        raise ValueError(
                            f"Expected a single attention mask, got {len(attention_mask)} instead, please open an "
                            "issue (https://github.com/huggingface/peft/issues) and report the error."
                        )
                    attention_mask = list(attention_mask.values())[0]

                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                if attention_mask.dim() == 4:
                    # Transform the 4d attention mask to 2d, leave it up to the model to deal with it instead of trying
                    # to create a 4d attention mask here.
                    # from [batch_size, heads, input_ids_length, total_sequence_length]
                    # to   [batch_size, total_sequence_length]
                    bs = attention_mask.shape[0]
                    total_seq_len = prefix_attention_mask.shape[1] + attention_mask.shape[2]
                    attention_mask_2d = torch.ones((bs, total_seq_len), dtype=attention_mask.dtype)

                    if is_prefill and (peft_config.peft_type != PeftType.PREFIX_TUNING):
                        # if in prefill stage, for prompt learning methods that are not prefix tuning, new tokens
                        # (embeddings) are inserted, thus set cache_position to correspond to these tokens
                        cache_position_ = torch.arange(total_seq_len, device=model_kwargs["input_ids"].device)
                    else:
                        # prefix tuning acts directly on the cache, no need to upate cache_position
                        cache_position_ = model_kwargs["cache_position"]

                    attention_mask_new = create_attention_mask(
                        self.get_base_model(),
                        model_input=None,
                        attention_mask=attention_mask_2d,
                        past_key_values=model_kwargs.get("past_key_values"),
                        cache_position=cache_position_,
                        batch_size=bs,
                        sequence_length=total_seq_len,
                        position_ids=model_kwargs.get("position_ids", None),
                    )
                    model_kwargs["attention_mask"] = attention_mask_new
                else:
                    # 2d attention mask
                    model_kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs.get("past_key_values", None) is None) or (
                isinstance(model_kwargs["past_key_values"], transformers.Cache)
                and not model_kwargs["past_key_values"].get_seq_length()
            )

            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                # some archs require max_cache_len to re-initialize the cache
                max_cache_len = getattr(model_kwargs.get("past_key_values", None), "max_cache_len", None)
                new_past_key_values = self.get_prompt(
                    batch_size=model_kwargs["input_ids"].shape[0],
                    max_cache_len=max_cache_len,
                )
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None

        # if we're in the prefill stage
        if is_prefill and (peft_config.peft_type == PeftType.PREFIX_TUNING):
            # for prefix tuning, the past_key_values have been prefilled
            model_kwargs["cache_position"] += peft_config.num_virtual_tokens
        elif peft_config.peft_type != PeftType.PREFIX_TUNING:  # prefix tuning needs cache_position
            # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is passed in the forward
            # pass to keep track of the position ids of the cache. We have to pop that from `model_kwargs` as
            # `cache_position` is properly created by the model, using the passed `inputs_embeds`:
            # https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
            _ = model_kwargs.pop("cache_position", None)

        return model_kwargs
import torch
from torch import nn
import peft
from peft import get_peft_model, PeftModelForCausalLM, PeftModel
import os

import time

from causal_mixed_peft import RecPeftMixedModelForCausalLM, PeftMixedModel


def create_recursive_peft_model(
    model,
    peft_config,
    recursion_depth=1,
    verbose=False
):

    device=model.device

    #get module list containing repeated transformer layers
    for name, module_list in model.named_modules():
        if isinstance(module_list, nn.ModuleList):
            break

    

    #print('1', (model.model.layers[0].self_attn.q_proj.weight- model.model.layers[1].self_attn.q_proj.weight).sum())
    #print(model.base_model.model.model.layers[0].self_attn.q_proj.road_alpha.default)
    
    #model = get_peft_model(model, peft_config, mixed=True).to(device)
    model = RecPeftMixedModelForCausalLM(model, peft_config, adapter_name='default').to(device)
    
    #print('2', (model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight- model.base_model.model.model.layers[1].self_attn.q_proj.base_layer.weight).sum())
    #peft_config.init_lora_weights="gaussian"
    
    #add recursion layers to model
    recursive_adapter_names = ['default']
    
    for rec_num in range(recursion_depth):
        rec_name = f"recursive_layer_{rec_num}"
        model.add_adapter(rec_name, peft_config)
        #recursive_adapter_names.append(rec_name)

    model.set_adapter(recursive_adapter_names) #activate 


    #for name, param in model.named_modules():
    #    if isinstance(param, nn.ModuleList):
    #        break

    
        
    wts = []
    replacement_name_list = []
    for transformer_blocknum,transformer_block in enumerate(module_list):
        block_wts = {}
        for blockname, block in transformer_block.named_modules():
            
            if isinstance(block,peft.tuners.lora.LoraLayer) or isinstance(block,peft.tuners.road.Linear):
                
                
                for layername, layer in block.named_modules():
                    if layername == 'lora_dropout':
                        continue
                    
                    if isinstance(layer,torch.nn.ModuleDict):
                        
                        for key,module in layer.items():
                            
                            if key=='default':
                                block_wts[blockname+layername+key+'weight'] = module.weight
                                if hasattr(module,'bias'):
                                    block_wts[blockname+layername+key+'bias'] = module.bias
                                continue
                            elif int(key[-1]) < transformer_blocknum:
                                if int(key[-1]) == 0:
                                    
                                    block_wts[blockname+layername+key+'weight'] = module.weight
                                    if hasattr(module,'bias'):
                                        block_wts[blockname+layername+key+'bias'] = module.bias
                                    module.weight = wts[-1][blockname+layername+'default'+'weight']
                                    if hasattr(module,'bias'):
                                        module.bias = wts[-1][blockname+layername+'default'+'bias']
                                    
                                else:
                                    block_wts[blockname+layername+key+'weight'] = module.weight
                                    if hasattr(module,'bias'):
                                        block_wts[blockname+layername+key+'bias'] = module.bias
                                    module.weight = wts[-1][blockname+layername+key[:-1]+str(int(key[-1])-1)+'weight']
                                    if hasattr(module,'bias'):
                                        module.bias = wts[-1][blockname+layername+key[:-1]+str(int(key[-1])-1)+'bias']
                                
                            else:
                                #module = torch.nn.Identity()
                                torch.nn.init.zeros_(module.weight)
                                if hasattr(module,'bias') and module.bias is not None:
                                    torch.nn.init.zeros_(module.bias)
                                module.requires_grad_(False)

                    if isinstance(layer,torch.nn.ParameterDict):
                        
                        for key,module in layer.items():
                            
                            if key=='default':
                                block_wts[blockname+layername+key] = module.data
                                
                                continue
                            elif int(key[-1]) < transformer_blocknum:
                                if int(key[-1]) == 0:
                                    
                                    block_wts[blockname+layername+key] = module.data
                                    module.data = wts[-1][blockname+layername+'default']
                                    #replacement_name_list.append([name+'['+str(transformer_blocknum-1)+'].'+blockname+'.'+layername+'.'+'default', 
                                    #                              name+'['+str(transformer_blocknum)+'].'+blockname+'.'+layername+'.'+ key])
                                    
                                    
                                    
                                else:
                                    block_wts[blockname+layername+key] = module.data
                                    module.data = wts[-1][blockname+layername+key[:-1]+str(int(key[-1])-1)]
                                    #replacement_name_list.append([name+'['+str(transformer_blocknum-1)+'].'+blockname+'.'+layername+'.'+ key[:-1]+str(int(key[-1])-1), 
                                    #                              name+'['+str(transformer_blocknum)+'].'+blockname+'.'+layername+'.'+ key])
                                    
                            else:
                                #module = torch.nn.Identity()
                                #torch.nn.init.zeros_(module)
                                module.data.zero_()
                                module.requires_grad_(False)

        wts.append(block_wts)
    

    
    
    return model


def load_recursive_peft_model(model, lora_dir):#,#torch_dtype=torch.float16,#device_map=None):
    #peft_model = PeftMixedModel.from_pretrained(model,lora_dir,init_weights=False)#,torch_dtype=torch_dtype,device_map=device_map).eval()
    peft_model = PeftModel.from_pretrained(model,lora_dir,torch_dtype=torch.float16)

    #peft_model.load_adapter(lora_dir,adapter_name='default')
    """
    rec_layers = sorted([x for x in os.listdir(lora_dir) if 'recursive_layer' in x])
    for adapter_name in rec_layers:
        res = peft_model.load_adapter(os.path.join(lora_dir,adapter_name),adapter_name=adapter_name,init_weights=False)
        #print(os.path.join(lora_dir,adapter_name))
        #print(res.missing_keys)
        
    
    rec_layers.insert(0,"default")
    #rec_layers = ['default']
    peft_model.set_adapter(rec_layers)
    #print(peft_model.active_adapter)
    """
    return peft_model
    
            
    
    



# 并行
## reduce的含义
![](20250702124040.png)
## batch 与 并行训练的关系
![](20250702124200.png)
## BN 与 并行训练的关系
![](20250702124737.png)
## 如何启动多个进程
### 方式一
![](20250702124805.png)

### 方式二
![](20250702124830.png)
## DistInfiniteBatchSampler中的dist针对的是torchrun还是numworker
![](20250708152652.png)
## 使用launch启动torchrun 只能调试子进程
![](20250802095825.png)
![](20250802095907.png)


# AI


## TCPstore
![](20250702124854.png)
## torch.empty(1).cuda().device
![](20250702125006.png)
![](20250702125016.png)
## cudnn库
![](20250703123624.png)
## 增大学习率k倍是否等价于loss放大k倍
![](20250703144316.png)
## weight decay
![](20250725131034.png)
## 层级学习率
![](20250725131229.png)
![](20250725131239.png)
## 梯度累积 accum_steps
![](20250726111856.png)
## 梯度裁剪 grad_clip
![](20250726111927.png)
![](20250806153228.png)
## RMSNorm
![](20250729202757.png)
![](20250729202806.png)
## extra_repr()
 ![](20250729203515.png)
## 反向传播
![](20250730111159.png)
![](20250730111210.png)
## KV_cache
https://zhuanlan.zhihu.com/p/662498827
![](20250824113900.png)
![](20250824113911.png)
## multi-head
![](20250730204423.png)
## flex_attn
https://pytorch.org/blog/flexattention/
## score
![](20250731175047.png)

# DistributedDataParallel
## device_ids=None参数行为分析
![](20250709125101.png)
## 单GPU模型，跨多个GPU分布模型
![](20250709125640.png)
## broadcast_buffers
![](20250709151204.png)
![](20250709151115.png)
![](20250709151128.png)
![](20250709151138.png)
## init_sync
![](20250709153156.png)
## process_group 
![](20250709160241.png)
![](20250709160249.png)


# 模型
## silu,sequential
![](20250708205243.png)
## unbind
![](20250708205257.png)
## FastRMSNorm vs nn.LayerNorm
![](20250708210938.png)
## norm_layer(C, elementwise_affine=False)
![](20250708211124.png)
![](20250708211148.png)
## self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)
## self.head_nm.ada_lin[-1].bias.data.zero_()
![](20250708212127.png)
![](20250708212153.png)
## nn.linear, bias
![](20250708212653.png)
![](20250708212722.png)
## 两种不同的模型参数初始化方法
![](20250708200245.png)
## 默认初始化方法是kaiming_uniform_ 
![](20250708234544.png)
## nn.parameters()与nn.modules()的区别
都是递归的
![](20250709112307.png)
![](20250709112315.png)
## count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
这是一个函数,接受参数m
![](20250709121849.png)
## TorchScript / ONNX
![](20250731110247.png)
![](20250731110301.png)
![](20250731110312.png)
## DropPath
 ![](20250801085105.png)
## CrossEntropyLoss
其中yi表示概率，最终loss越低越好
![](20250817154709.png)

# Python知识

## pyi文件的作用
![](20250702131230.png)
## python vscode 调试默认不进入第三方代码
![](20250702160737.png)
## 函数参数默认值
![](20250702161559.png)
## Optional的优点
![](20250702161708.png)
## stdout与stderr
![](20250703121552.png)
## torch检测反向梯度错误
![](20250703122443.png)
## or
![](20250703144437.png)
## 浮点数转字符串
![](20250703150430.png)
## re.compile(r'[^\w\-+,.]')的用法
![](20250703150801.png)
![](20250703150814.png)
## sorted(glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True)
![](20250703151536.png)
## tb_lg: misc.TensorboardLogger
![](20250703153337.png)
## get_attr
![](20250703160422.png)
## disklogger的作用
![](20250703181645.png)
## __call__
![](20250703183316.png)
## mode类型
mode = {
    1: 'reduce-overhead',
    2: 'max-autotune',
    3: 'default',
}[fast]
的结果类型是字符串（str）。
## partial
![](20250708210915.png)
## Union
![](20250708233027.png)
## for block_idx, sab in enumerate(self.unregistered_blocks):
block_idx是从0开始
## sab.sa.proj.weight.data.mul_(scale)
![](20250708233836.png)
## @staticmethod
![](20250730104021.png)
![](20250730104032.png)
## @classmethod
![](20250730104101.png)
## [::-1]
![](20250804173924.png)
## lambda lamuda
![](20250804180948.png)

# torch计算

## torch乘法精度
![](20250703122929.png)
## 梯度积累
![](20250703125808.png)
![](20250703125817.png)
## state_dict()
![](20250708163542.png)
## 模型checkpoint两种形式
![](20250708163724.png)
![](20250708163738.png)
## 计算图checkpoint原理
![](20250703145452.png)
## ckpt中的ep和it
![](20250703152459.png)
## torch.compile(加速推理)
![](20250708194445.png)
![](20250708194520.png)
## torch.cuda.synchronize()
![](20250709153846.png)
## named_parameters
![](20250709155627.png)
## ndim
![](20250709155803.png)
## tensor.item()
注：只有shape为0的时候才可以用item()
![](20250725164251.png)
## torch.cuda.amp.autocast()
![](20250726103838.png)
## torch.cuda.amp.GradScaler()
![](20250726104053.png)
![](20250726112232.png)
##  for t in optimizer_state['found_inf_per_device'].values(): dist.allreduce(t)
![](20250726165900.png)
![](20250726165924.png)
![](20250726165958.png)
## allreduce
![](20250726170205.png)
![](20250726170210.png)
## f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
![](20250728164128.png)
## expand
![](20250729094313.png)
## torch.full
![](20250729095614.png)
## contiguous()
![](20250729100722.png)
## [:, 0]
![](20250729100831.png)
## register_buffer()
![](20250729101109.png)
## x_BLC.new_ones(8, 8)
![](20250729101323.png)
## nn.embedding
![](20250729105051.png)
## pad
![](20250729160731.png)
## dropout
![](20250731092945.png)
![](20250731093007.png)
## optimizer.load_state_dict()
![](20250731105827.png)
## unbind
![](20250731153151.png)
## clamp
具体来说，clamp(min=1e-12) 会将 cdf_plus 中所有小于 1e-12（即 0.000000000001）的值都替换为 1e-12，而大于或等于这个值的元素则保持不变。
## mean
![](20250804175501.png)
## nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')  这里的label_smoothing是啥意思
![](20250804185544.png)
![](20250804185614.png)
## bincount 
![](20250806154413.png)
## unsqueeze(dim)
![](20250824110017.png)


# dataloder
## sampler和batch_sampler的区别
![](20250707190717.png)
![](20250707190726.png)
## sampler举例
![](20250708121226.png)
## sampler中的__len__是什么意思
我们可以得出结论,len(sampler)就是一个epoch中的iteration个数
![](20250708130253.png)
## collate_fn
![](20250707192318.png)
![](20250707193622.png)
## default_collate(),也就是默认的collate_fn
![](20250707192354.png)
![](20250707192404.png)
![](20250707192412.png)
## dataloder 会自动执行 pin_memory吗
![](20250707195137.png)
## timeout参数一旦超过时间就报错退出
![](20250707195336.png)
## generator参数
![](20250707204441.png)
![](20250707204452.png)
## torch.generator()
![](20250708121923.png)
## 每个worker（dataloder中的worker）是否处理一小部分batch，然后再拼起来
![](20250707212940.png)
## prefetch_factor
prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
    in advance by each worker. ``2`` means there will be a total of
    2 * num_workers batches prefetched across all workers. (default value depends
    on the set value for num_workers. If value of num_workers=0 default is ``None``.
    Otherwise, if value of ``num_workers > 0`` default is ``2``).
![](20250707213100.png)
## persistent_workers 
(bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
## pin_memory_device
![](20250707213646.png)
## in_order
![](20250707213833.png)

# python计算

## tuple
![](20250703124537.png)
实际：
![](20250703124742.png)
## __iter__()
![](20250708121409.png)
## list与__iter__()
![](20250708121438.png)
## indices[:tails]
![](20250708132538.png)
## 迭代器的使用,两种方法:for与iter
![](20250708161053.png)
## opt_kw = dict(lr=args.tlr, weight_decay=0)
![](20250725155146.png)
## defaultdict(SmoothedValue)
![](20250725164808.png)
## fmt.format
![](20250725165543.png)
## datetime.timedelta(seconds=...)
![](20250725174831.png)
## eta是什么意思
![](20250725174911.png)
## with
![](20250728184232.png)
![](20250728184432.png)
## /
![](20250801161902.png)
## def continuous_gaussian_log_likelihood(x, *, means, log_scales):
在 Python 函数定义中，参数列表中的 * 有一个特殊的作用，它表示 * 之后的所有参数都必须以 关键字参数（keyword arguments）的形式传递，而不能以位置参数（positional arguments）的形式传递。
## kw = dict(z_voc_usage=cluster_usage)
kw = {"z_voc_usage": cluster_usage}
## round(x)
四舍五入
## str(datetime.timedelta(seconds=...))
![](20250817154110.png)
## time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))
![](20250817154242.png)
## _reg_valid_name.sub
![](20250817175022.png)


# 面向对象

## 多态
![](20250708131917.png)
## 子类调用了父类的方法，而父类的方法中用到了 self.xxx 这样的成员变量
![](20250708150042.png)
## 父类初始化
![](20250708150118.png)
## x = torch.tensor([[1, 2],[3, 4]])形状
torch.Size([2, 2])
## repeat_interleave
![](20250725165659.png)
![](20250725165722.png)
## @property
![](20250725165509.png)
## shutil.copy(local_out_ckpt, local_out_ckpt_best)
![](20250817170322.png)



# np

## np.linspace
np.linspace(2.0, 3.0, num=5)
array([2.  , 2.25, 2.5 , 2.75, 3.  ])
np.linspace(2.0, 3.0, num=5, endpoint=False)
array([2. ,  2.2,  2.4,  2.6,  2.8])
np.linspace(2.0, 3.0, num=5, retstep=True)
(array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
## np.prod
![](20250801164635.png)
## np.cumprod
![](20250803163148.png)

# var

## 文件

print在misc.py文件里

if args.pg:  # 只有设置了 progressive training（pg > 0.0）才会进入

g_it 全局步数

stepping 当前轮是否执行反向更新，这个参数与梯度累积 accum_steps有关

## 变量

## f, f_hat
注意 f_hat不是index，而是embedding
![](20250728104753.png)
## f_hat_or_idx_Bl
![](20250728113035.png)
## gt_idx_Bl_super 同时也是 f_hat_or_idx_Bl
![](20250728113354.png)
## f_hat_super
shape = torch.Size([4, 32, 16, 16])
## gt_BL_super
torch.Size([4, 680])
## quant_resi模块
![](20250728115151.png)
## idxBl_to_var_input函数
用来获取var的input
![](20250728115731.png)
## x_BLCv_wo_first_l_super
shape = torch.Size([4, 679, 32])
## scale_schedule
![](20250728185948.png)
## torch.tensor([lowLen] * B, dtype=torch.int32)
![](20250729113200.png)
## label_B
torch.Size([4]) 表示标签
## need_to_pad
![](20250729160934.png)
## shared_ada_lin
![](20250731112322.png)
## act, ada_lin, gss
act: activation
ada_lin: Adaptive LayerNorm Parameter Linear Mapper
ada_gss: gamma, scale, shift
## shared_aln
用于控制是否 共享一组 LayerNorm 参数（γ/β），而不是对每个样本通过条件向量单独生成。
![](20250731152147.png)
## approx_standard_normal_cdf
这是一个用于快速近似计算标准正态分布的 累积分布函数（Cumulative Distribution Function, CDF）的函数。
## def continuous_gaussian_log_likelihood(x, *, means, log_scales):
![](20250802152038.png)
## def normal_kl(mean1, logvar1, mean2, logvar2):
![](20250802152209.png)
![](20250802152343.png)
## metric
| 名称 | 作用 |
| --- |  --- | 
| `tlr` total learning rate | 最大学习率 | 
| `tnm` Total Norm | = grad_norm， 所有梯度的L2范式 | 
| `Lm`  | logits_BLV和gt_BL_super的交叉熵，logits_BLV.shape = （B，L，V） | 
| `Lt` | tail的交叉熵 | 
| `Accm` | 所有token的正确率， acc_mean = (pred_BL == gt_BL_super).float().mean().item() * 100 # int | 
| `Acct` | tail（也就是最后一层 16 * 16）token的正确率 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
| `tnm` | = grad_norm， 所有梯度的L2范式 | 
## low_proj_for_sos
![](20250823214045.png)
## TextAttentivePool
![](20250823214131.png)


# diffusion
## betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
![](20250803151505.png)
![](20250803152247.png)
## def space_timesteps(num_timesteps, section_counts): 
![](20250803152117.png)
## def create_diffusion
timestep_respacing : section_counts
## timestep_map : LEARNED_RANGE
![](20250803153142.png)
## use_timesteps vs timestep_map
![](20250803162844.png)
## new_betas
new_betas[1 - alpha_cumprod / last_alpha]
## model_mean_type
EPSILON	模型预测的是添加的噪声 𝜖
START_X	模型直接预测原始图像  $ x_{0} $​
## betas
shape(1000, )
type: ndarray
## posterior_variance
posterior_log_variance_clipped
posterior_mean_coef1
posterior_mean_coef2
![](20250803165440.png)
## q_mean_variance
![](20250803180811.png)
## _extract_into_tensor(...)
![](20250803180842.png)
## q_sample
sample from q(x_t | x_0).
return x_t
## signal
:param model: the model, which takes a signal and a batch of timesteps as input.
这里的 signal 是指：当前的扩散状态张量x_t，也就是模型在某个时间步 t 上接收到的图像或特征输入。
## p_mean_variance(...)
![](20250804110349.png)
![](20250804110400.png)
xt->x0->q_posterior_mean_variance.mean
if learned_var: xt->var
else q_posterior_mean_variance.var
## _predict_xstart_from_eps,_predict_eps_from_xstart
![](20250804112507.png)
![](20250804112520.png)
## condition_mean
![](20250804113024.png)
## condition_score
![](20250804122052.png)
![](20250804122103.png)
xt->eps->x0->q_posterior_mean_variance.mean
## ddim_sample and ddim_reverse_sample
| `ddim_sample` | 生成图像 | 正向推理（x\_T → x\_0） | 从随机噪声逐步去噪，生成图像 |
| --- |  --- |  --- |  --- |

| `ddim_reverse_sample` | 编码图像 | 反向推理（x\_0 → x\_T） | 从原始图像合成噪声轨迹，用于训练、重建或反向调控 |
| --- |  --- |  --- |  --- |
## DDIM
https://zhuanlan.zhihu.com/p/614147698

# tensorboard
## events.out.tfevents.{时间戳}.{主机名}.{进程ID}{filename_suffix}
![](20250817194053.png)
## tensorboard --logdir .
或者local_output/__b4ep50adamlr0.0003wd0.005
## ps aux | grep tensorboard
## pkill -f tensorboard
## pkill -f tensorboard_data_server
## x = np.random.random(1000)
生成一个 长度为 1000 的数组，元素是服从 [0,1) 区间均匀分布

# distracted

B站视频，吃饭，聊天，想打开游戏练枪/设置快捷键,看qq


# 看不懂
## scale_schedule
![](20250729111602.png)
## progressive training
in var-origin
![](20250729111703.png)
## use_flex_attn

## attn_bias_or_two_vector

# SRVAR
## SRVAR::forward
inp_B3HW_low : [B, 3, H, W]
low_f : [B, 3, h, w] -> [B, C, h1, w1] using vartokenizer maybe
low_f : [B, h1*w1, C]

if(ref) low_f = [B, 2*h1*w1, C]

lowlen, lowC = 2*h1*w1, C
lens.shape = [B] 
max_seqlen_k = max lens
cu_seqlens_k = cumsum((0, lens))
kv_compact = [B, 2*h1*w1, C] = low_f
kv_compact = [B*2*h1*21, C]
kv_compact = [someplace(in lowlen unit) replaced by cfg_uncond] in dimension 0
cond_BD = sos = low_proj_for_sos(kv_compact) [B, C2] in this case C2 = 1024
cond_BD_or_gss = shared_ada_lin(cond_BD) # gss: gamma, scale, shift;torch.Size([4, 1024])

this thing later translate to [B, 6*C] through self.ada_lin in basic.py in line 517

sos : [B, C2]->[B, 1, C2]
x_BLC = (sos, x_BLC_wo_prefix) : [B, 680, C2]
l_end = 680
attn_bias_for_masking : [1, 1, l_end, l_end] this is masking

x_BLC->transformer->x_BLC
x_BLC : [B, 680, C2] -> [B, 680, C3] in this case C3 = 4096
return x_BLC

## SRtrainer::train_step

input : inp_B3HW_super : [B, 3, H, W]
gt_idx_Bl_super : img_to_idxBl(inp_B3HW_super) : ([B, 1], [B, 4], ... , [B, 256])
gt_BL_super = cat(gt_idx_Bl_super) : [B, 680]
x_BLCv_wo_first_l_super : idxBl_to_var_input(gt_idx_Bl_super) # torch.Size([4, 679, 32])


logits_BLV, diff_loss = SRVAR::forward(inp_B3HW_low, x_BLCv_wo_first_l_super)

loss = train_loss(logits_BLV.view(B * 680, C3), gt_BL_super.view(-1))

so this most important point is the two function:
1. img_to_idxBl
2. idxBl_to_var_input

## img_to_idxBl
## idxBl_to_var_input

## SRVAR::autoregressive_infer_cfg
low_f = encode(inp_B3HW_low) [1, 32, 16, 16] 
low_f : [1, 32, 16, 16] -> [1, 256, 32]

in the following context, B = 1
lowlen, lowC = 2*h1*w1, C
lens.shape = [B] 
max_seqlen_k = max lens
cu_seqlens_k = cumsum((0, lens))
kv_compact = [B, 2*h1*w1, C] = low_f
kv_compact = [B*2*h1*21, C]
kv_compact = [someplace(in lowlen unit) replaced by cfg_uncond] in dimension 0
cond_BD = sos = low_proj_for_sos(kv_compact) [B, C2] in this case C2 = 1024 

cond_BD_or_gss = shared_ada_lin(cond_BD) # gss: gamma, scale, shift;torch.Size([B, 1024]) 这里的B也是1
accu_BChw, cur_L, ret = None, 0, []  
idx_Bl_list = []

accu_BChw : [1, 32, 16, 16] all zero





## eval.ipynb

gt_idx_Bl_super : img_to_idxBl(inp_B3HW_super) : ([B, 1], [B, 4], ... , [B, 256])
gt_BL_super = cat(gt_idx_Bl_super) : [B, 680]
x_BLCv_wo_first_l_super : idxBl_to_var_input(gt_idx_Bl_super) # torch.Size([4, 679, 32])

ret, idx_Bl_list, img = srvar.autoregressive_infer_cfg(...)



# MAR

## forward_mae_encoder
input x : [B, L, C], mask : [B, L], class_embedding : [B, C]
x, mask : [b, L, c] -> [b, L + buffer_size, C]
x[:, :self.buffer_size] = class_embedding
x = x + pos_embed
x : [B, L, c] -> [B, L2, C] (mask, 0 是有效的也就是没有被遮住的)
x -> transformer -> x
return x

## forward_mar_decoder
x : [B, L2, C], mask : [B, L]
mask : [B, L] -> [B, L + buffer_size]
mask_tokens : [1, 1, C] -> [B, L + buffer_size, C]
x_after_pad : [B, L + buffer_size, C] -> [B, L + buffer_size, C] (将 x 的值传递给 x_after_pad中为0 的部分， 长度应该刚刚好)
x = x_after_pad + self.decoder_pos_embed_learned
x -> transformer -> x
x = x[:, self.buffer_size:]
x = x + self.diffusion_pos_embed_learned
return x

## forward_loss
input : z, mask, target : [B, L, C]
diffusion_batch_mul seems like 4?
diffusion_batch_mul = dbm
target : [B * L, C] -> [dbm * B * L, C]
z : [B * L, C] -> [dbm * B * L, C]
mask : [B * L, C] -> [dbm * B * L, C]
loss = diffloss(z, target, mask)
return loss

## DiffLoss::forward
input : target, z, mask : [B, L, C]
t : [B]
loss_dict = training_losses(self.net, target, t, model_kwargs)
loss : [B, L]
loss = (loss * mask) (相当于mask 为 1才是需要预测，计算loss的)
loss = loss.sum() / mask.sum()
return loss.mean()

## sample_tokens
input : bsz = len(labels), num_iter, cfg_scale, cfg_schedule, labels : tuple，其中每一个元素代表想生成的类型, temperature, progress : True(是否进度条)

mask : [b, L] ones
tokens : [b, L, C] zeros
orders : [b, C]
indices : [0, 1, 2, ... , num_iter - 1]
for step in indices:
    cur_tokens = tokens
    class_embedding = [bsz, C]
    tokens = tokens, tokens [2 * B, L]
    class_embedding = class_embedding, fake_latent [2 * B, C]
    mask = mask, mask [2 * B, L]

    x = forward_mae_encoder(tokens, mask, class_embedding) [2 * B, L1, C]
    z = forward_mae_decoder(x, mask) [2 * B, L, C]

    mask_ratio : smaller with step increase
    mask_len : mask_ratio * L
    // torch.sum(mask, dim=-1, keepdims=True)是最后一维变1的意思
    mask_len = max(1, min(len - 1, mask_len)) 其中len为 mask 中 1 的个数，也就是被遮掩 [B]
    
    def mask_by_order:
        masking : [B, L] zeros
        // torch.scatter(masking, dim=0, index=index, src=src), dim=0 的意思就是index填充的是dim=0的数据，也就是说index列的数量一定要和masking是一样的      
        masking : [B, mask]对应地方变1，每个batch有 mask_len这么多个1，表示被遮掩

    mask_next = mask_by_order(mask_len[0], orders, bsz, L)，这里注意orders是循环外定义的，所以不会出现之前未被mask的现在mask了，一定是按照顺序的


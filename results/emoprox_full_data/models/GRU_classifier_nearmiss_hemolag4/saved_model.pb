­2
Á  
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Î®0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

: *
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

gru/gru_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬`*&
shared_namegru/gru_cell_2/kernel

)gru/gru_cell_2/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/kernel*
_output_shapes
:	¬`*
dtype0

gru/gru_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*0
shared_name!gru/gru_cell_2/recurrent_kernel

3gru/gru_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/recurrent_kernel*
_output_shapes

: `*
dtype0

gru/gru_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*$
shared_namegru/gru_cell_2/bias
{
'gru/gru_cell_2/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/bias*
_output_shapes

:`*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

: *
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0

Adam/gru/gru_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬`*-
shared_nameAdam/gru/gru_cell_2/kernel/m

0Adam/gru/gru_cell_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell_2/kernel/m*
_output_shapes
:	¬`*
dtype0
¨
&Adam/gru/gru_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*7
shared_name(&Adam/gru/gru_cell_2/recurrent_kernel/m
¡
:Adam/gru/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/gru/gru_cell_2/recurrent_kernel/m*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*+
shared_nameAdam/gru/gru_cell_2/bias/m

.Adam/gru/gru_cell_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell_2/bias/m*
_output_shapes

:`*
dtype0

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

: *
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

Adam/gru/gru_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬`*-
shared_nameAdam/gru/gru_cell_2/kernel/v

0Adam/gru/gru_cell_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell_2/kernel/v*
_output_shapes
:	¬`*
dtype0
¨
&Adam/gru/gru_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*7
shared_name(&Adam/gru/gru_cell_2/recurrent_kernel/v
¡
:Adam/gru/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/gru/gru_cell_2/recurrent_kernel/v*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*+
shared_nameAdam/gru/gru_cell_2/bias/v

.Adam/gru/gru_cell_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell_2/bias/v*
_output_shapes

:`*
dtype0

NoOpNoOp
¤%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ß$
valueÕ$BÒ$ BË$
Ù
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratemLmM mN!mO"mPvQvR vS!vT"vU
 
#
 0
!1
"2
3
4
#
 0
!1
"2
3
4
­
#non_trainable_variables
$metrics
regularization_losses
trainable_variables

%layers
&layer_regularization_losses
'layer_metrics
	variables
 
 
 
 
­
(metrics
)non_trainable_variables
regularization_losses
trainable_variables

*layers
+layer_regularization_losses
,layer_metrics
	variables
~

 kernel
!recurrent_kernel
"bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
 
 

 0
!1
"2

 0
!1
"2
¹
1non_trainable_variables
2metrics
regularization_losses
trainable_variables

3layers
4layer_regularization_losses

5states
6layer_metrics
	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
7metrics
8non_trainable_variables
regularization_losses
trainable_variables

9layers
:layer_regularization_losses
;layer_metrics
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEgru/gru_cell_2/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEgru/gru_cell_2/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell_2/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

0
1
2
3
 
 
 
 
 
 
 
 

 0
!1
"2

 0
!1
"2
­
>metrics
?non_trainable_variables
-regularization_losses
.trainable_variables

@layers
Alayer_regularization_losses
Blayer_metrics
/	variables
 
 

0
 
 
 
 
 
 
 
 
4
	Ctotal
	Dcount
E	variables
F	keras_api
D
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

J	variables
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/gru/gru_cell_2/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/gru/gru_cell_2/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/gru/gru_cell_2/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/gru/gru_cell_2/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/gru/gru_cell_2/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/gru/gru_cell_2/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_inputPlaceholder*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
dtype0**
shape!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
´
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputgru/gru_cell_2/biasgru/gru_cell_2/kernelgru/gru_cell_2/recurrent_kerneloutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8 *,
f'R%
#__inference_signature_wrapper_45753
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ý	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)gru/gru_cell_2/kernel/Read/ReadVariableOp3gru/gru_cell_2/recurrent_kernel/Read/ReadVariableOp'gru/gru_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp0Adam/gru/gru_cell_2/kernel/m/Read/ReadVariableOp:Adam/gru/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOp.Adam/gru/gru_cell_2/bias/m/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOp0Adam/gru/gru_cell_2/kernel/v/Read/ReadVariableOp:Adam/gru/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOp.Adam/gru/gru_cell_2/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *'
f"R 
__inference__traced_save_48499

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell_2/kernelgru/gru_cell_2/recurrent_kernelgru/gru_cell_2/biastotalcounttotal_1count_1Adam/output/kernel/mAdam/output/bias/mAdam/gru/gru_cell_2/kernel/m&Adam/gru/gru_cell_2/recurrent_kernel/mAdam/gru/gru_cell_2/bias/mAdam/output/kernel/vAdam/output/bias/vAdam/gru/gru_cell_2/kernel/v&Adam/gru/gru_cell_2/recurrent_kernel/vAdam/gru/gru_cell_2/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 **
f%R#
!__inference__traced_restore_48581Å/
°

×
*__inference_gru_cell_2_layer_call_fn_48092

inputs
states_0
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity

identity_1¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *N
fIRG
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_440942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
õ
C
'__inference_masking_layer_call_fn_46588

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_447912
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ô
Ú
>__inference_gru_layer_call_and_return_conditional_losses_47636

inputs4
"gru_cell_2_readvariableop_resource:`7
$gru_cell_2_readvariableop_1_resource:	¬`6
$gru_cell_2_readvariableop_4_resource: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell_2/ReadVariableOp¢gru_cell_2/ReadVariableOp_1¢gru_cell_2/ReadVariableOp_2¢gru_cell_2/ReadVariableOp_3¢gru_cell_2/ReadVariableOp_4¢gru_cell_2/ReadVariableOp_5¢gru_cell_2/ReadVariableOp_6¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2
gru_cell_2/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like/Const±
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/ones_likez
gru_cell_2/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like_1/Shape
gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like_1/Const¸
gru_cell_2/ones_like_1Fill%gru_cell_2/ones_like_1/Shape:output:0%gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/ones_like_1
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_2/ReadVariableOp
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_2/unstack
gru_cell_2/mulMulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul
gru_cell_2/mul_1Mulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_1
gru_cell_2/mul_2Mulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_2 
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_1
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice/stack_1
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2Á
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice
gru_cell_2/MatMulMatMulgru_cell_2/mul:z:0!gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul 
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_2
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_1/stack
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_1/stack_1
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2Ë
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_1¡
gru_cell_2/MatMul_1MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_1 
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_3
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell_2/strided_slice_2/stack
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2Ë
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_2¡
gru_cell_2/MatMul_2MatMulgru_cell_2/mul_2:z:0#gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_2
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_3/stack_1
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2®
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_3§
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_4/stack
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_4/stack_1
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_4­
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_1
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell_2/strided_slice_5/stack
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2¬
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_5­
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_2
gru_cell_2/mul_3Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_3
gru_cell_2/mul_4Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_4
gru_cell_2/mul_5Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_5
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_4
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_6/stack_1
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2Ê
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_6¡
gru_cell_2/MatMul_3MatMulgru_cell_2/mul_3:z:0#gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_3
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_5
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_7/stack
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_7/stack_1
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2Ê
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_7¡
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_4:z:0#gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_4
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_8/stack_1
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2®
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_8­
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_3
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_9/stack
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_9/stack_1
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_9­
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_4
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/addy
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_1
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid_1
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_6
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!gru_cell_2/strided_slice_10/stack
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2Ï
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_10¢
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_5:z:0$gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_5
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2#
!gru_cell_2/strided_slice_11/stack
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2±
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_11®
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_5
gru_cell_2/mul_6Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_6
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_2r
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Tanh
gru_cell_2/mul_7Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_7i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/sub/x
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/sub
gru_cell_2/mul_8Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_8
gru_cell_2/add_3AddV2gru_cell_2/mul_7:z:0gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_47472*
condR
while_cond_47471*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulë
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÆ
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Â 
ø
A__inference_output_layer_call_and_return_conditional_losses_48066

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd¥
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª

Ë
gru_while_cond_45929$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_45929___redundant_placeholder0;
7gru_while_gru_while_cond_45929___redundant_placeholder1;
7gru_while_gru_while_cond_45929___redundant_placeholder2;
7gru_while_gru_while_cond_45929___redundant_placeholder3;
7gru_while_gru_while_cond_45929___redundant_placeholder4
gru_while_identity

gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
õ
¥
while_cond_47814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47814___redundant_placeholder03
/while_while_cond_47814___redundant_placeholder13
/while_while_cond_47814___redundant_placeholder23
/while_while_cond_47814___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


E__inference_gru_cell_2_layer_call_and_return_conditional_losses_48220

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ones_like_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ÿ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ì
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2Ú
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2ê
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_2g
mul_3Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5~
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_3~
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2ì
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2Ú
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stack
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2ï
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3Í
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulà
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÞ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityâ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
õ
¥
while_cond_47128
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47128___redundant_placeholder03
/while_while_cond_47128___redundant_placeholder13
/while_while_cond_47128___redundant_placeholder23
/while_while_cond_47128___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
õ
¥
while_cond_47471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47471___redundant_placeholder03
/while_while_cond_47471___redundant_placeholder13
/while_while_cond_47471___redundant_placeholder23
/while_while_cond_47471___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ð
Ú
>__inference_gru_layer_call_and_return_conditional_losses_45571

inputs4
"gru_cell_2_readvariableop_resource:`7
$gru_cell_2_readvariableop_1_resource:	¬`6
$gru_cell_2_readvariableop_4_resource: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell_2/ReadVariableOp¢gru_cell_2/ReadVariableOp_1¢gru_cell_2/ReadVariableOp_2¢gru_cell_2/ReadVariableOp_3¢gru_cell_2/ReadVariableOp_4¢gru_cell_2/ReadVariableOp_5¢gru_cell_2/ReadVariableOp_6¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2
gru_cell_2/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like/Const±
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/ones_likey
gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout/Const¬
gru_cell_2/dropout/MulMulgru_cell_2/ones_like:output:0!gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Mul
gru_cell_2/dropout/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout/Shapeò
/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ñ²º21
/gru_cell_2/dropout/random_uniform/RandomUniform
!gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell_2/dropout/GreaterEqual/yë
gru_cell_2/dropout/GreaterEqualGreaterEqual8gru_cell_2/dropout/random_uniform/RandomUniform:output:0*gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell_2/dropout/GreaterEqual¡
gru_cell_2/dropout/CastCast#gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Cast§
gru_cell_2/dropout/Mul_1Mulgru_cell_2/dropout/Mul:z:0gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Mul_1}
gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_1/Const²
gru_cell_2/dropout_1/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Mul
gru_cell_2/dropout_1/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_1/Shapeø
1gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2òË23
1gru_cell_2/dropout_1/random_uniform/RandomUniform
#gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_1/GreaterEqual/yó
!gru_cell_2/dropout_1/GreaterEqualGreaterEqual:gru_cell_2/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru_cell_2/dropout_1/GreaterEqual§
gru_cell_2/dropout_1/CastCast%gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Cast¯
gru_cell_2/dropout_1/Mul_1Mulgru_cell_2/dropout_1/Mul:z:0gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Mul_1}
gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_2/Const²
gru_cell_2/dropout_2/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Mul
gru_cell_2/dropout_2/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_2/Shapeø
1gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2È«¡23
1gru_cell_2/dropout_2/random_uniform/RandomUniform
#gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_2/GreaterEqual/yó
!gru_cell_2/dropout_2/GreaterEqualGreaterEqual:gru_cell_2/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru_cell_2/dropout_2/GreaterEqual§
gru_cell_2/dropout_2/CastCast%gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Cast¯
gru_cell_2/dropout_2/Mul_1Mulgru_cell_2/dropout_2/Mul:z:0gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Mul_1z
gru_cell_2/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like_1/Shape
gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like_1/Const¸
gru_cell_2/ones_like_1Fill%gru_cell_2/ones_like_1/Shape:output:0%gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/ones_like_1}
gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_3/Const³
gru_cell_2/dropout_3/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Mul
gru_cell_2/dropout_3/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_3/Shape÷
1gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2´Á23
1gru_cell_2/dropout_3/random_uniform/RandomUniform
#gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_3/GreaterEqual/yò
!gru_cell_2/dropout_3/GreaterEqualGreaterEqual:gru_cell_2/dropout_3/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_3/GreaterEqual¦
gru_cell_2/dropout_3/CastCast%gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Cast®
gru_cell_2/dropout_3/Mul_1Mulgru_cell_2/dropout_3/Mul:z:0gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Mul_1}
gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_4/Const³
gru_cell_2/dropout_4/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Mul
gru_cell_2/dropout_4/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_4/Shape÷
1gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2úÑÈ23
1gru_cell_2/dropout_4/random_uniform/RandomUniform
#gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_4/GreaterEqual/yò
!gru_cell_2/dropout_4/GreaterEqualGreaterEqual:gru_cell_2/dropout_4/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_4/GreaterEqual¦
gru_cell_2/dropout_4/CastCast%gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Cast®
gru_cell_2/dropout_4/Mul_1Mulgru_cell_2/dropout_4/Mul:z:0gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Mul_1}
gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_5/Const³
gru_cell_2/dropout_5/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Mul
gru_cell_2/dropout_5/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_5/Shapeö
1gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2íA23
1gru_cell_2/dropout_5/random_uniform/RandomUniform
#gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_5/GreaterEqual/yò
!gru_cell_2/dropout_5/GreaterEqualGreaterEqual:gru_cell_2/dropout_5/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_5/GreaterEqual¦
gru_cell_2/dropout_5/CastCast%gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Cast®
gru_cell_2/dropout_5/Mul_1Mulgru_cell_2/dropout_5/Mul:z:0gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Mul_1
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_2/ReadVariableOp
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_2/unstack
gru_cell_2/mulMulstrided_slice_2:output:0gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul
gru_cell_2/mul_1Mulstrided_slice_2:output:0gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_1
gru_cell_2/mul_2Mulstrided_slice_2:output:0gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_2 
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_1
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice/stack_1
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2Á
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice
gru_cell_2/MatMulMatMulgru_cell_2/mul:z:0!gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul 
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_2
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_1/stack
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_1/stack_1
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2Ë
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_1¡
gru_cell_2/MatMul_1MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_1 
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_3
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell_2/strided_slice_2/stack
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2Ë
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_2¡
gru_cell_2/MatMul_2MatMulgru_cell_2/mul_2:z:0#gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_2
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_3/stack_1
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2®
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_3§
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_4/stack
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_4/stack_1
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_4­
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_1
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell_2/strided_slice_5/stack
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2¬
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_5­
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_2
gru_cell_2/mul_3Mulzeros:output:0gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_3
gru_cell_2/mul_4Mulzeros:output:0gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_4
gru_cell_2/mul_5Mulzeros:output:0gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_5
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_4
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_6/stack_1
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2Ê
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_6¡
gru_cell_2/MatMul_3MatMulgru_cell_2/mul_3:z:0#gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_3
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_5
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_7/stack
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_7/stack_1
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2Ê
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_7¡
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_4:z:0#gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_4
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_8/stack_1
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2®
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_8­
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_3
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_9/stack
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_9/stack_1
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_9­
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_4
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/addy
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_1
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid_1
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_6
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!gru_cell_2/strided_slice_10/stack
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2Ï
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_10¢
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_5:z:0$gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_5
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2#
!gru_cell_2/strided_slice_11/stack
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2±
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_11®
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_5
gru_cell_2/mul_6Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_6
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_2r
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Tanh
gru_cell_2/mul_7Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_7i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/sub/x
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/sub
gru_cell_2/mul_8Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_8
gru_cell_2/add_3AddV2gru_cell_2/mul_7:z:0gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_45359*
condR
while_cond_45358*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulë
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÆ
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs

´
#__inference_gru_layer_call_fn_46622
inputs_0
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_441832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
õ
¥
while_cond_44438
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_44438___redundant_placeholder03
/while_while_cond_44438___redundant_placeholder13
/while_while_cond_44438___redundant_placeholder23
/while_while_cond_44438___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
à'
¸
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45689	
input
	gru_45664:`
	gru_45666:	¬`
	gru_45668: `
output_45671: 
output_45673:
identity¢gru/StatefulPartitionedCall¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallâ
masking/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_447912
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_45664	gru_45666	gru_45668*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_450872
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_45671output_45673*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_451252 
output/StatefulPartitionedCall½
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45666*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulÐ
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45668*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall8^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
Øá
Å
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_46583

inputs8
&gru_gru_cell_2_readvariableop_resource:`;
(gru_gru_cell_2_readvariableop_1_resource:	¬`:
(gru_gru_cell_2_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identity¢gru/gru_cell_2/ReadVariableOp¢gru/gru_cell_2/ReadVariableOp_1¢gru/gru_cell_2/ReadVariableOp_2¢gru/gru_cell_2/ReadVariableOp_3¢gru/gru_cell_2/ReadVariableOp_4¢gru/gru_cell_2/ReadVariableOp_5¢gru/gru_cell_2/ReadVariableOp_6¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢	gru/while¢output/BiasAdd/ReadVariableOp¢output/Tensordot/ReadVariableOpm
masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking/NotEqual/y
masking/NotEqualNotEqualinputsmasking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
masking/NotEqual
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
masking/Any/reduction_indices¦
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
masking/Any
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
masking/mul
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
masking/SqueezeU
	gru/ShapeShapemasking/mul:z:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ú
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm
gru/transpose	Transposemasking/mul:z:0gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1s
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/ExpandDims/dim¤
gru/ExpandDims
ExpandDimsmasking/Squeeze:output:0gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gru/ExpandDims
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm¦
gru/transpose_1	Transposegru/ExpandDims:output:0gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gru/transpose_1
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
gru/TensorArrayV2/element_shapeÂ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2Ç
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
gru/strided_slice_2
gru/gru_cell_2/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/ones_like/Shape
gru/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
gru/gru_cell_2/ones_like/ConstÁ
gru/gru_cell_2/ones_likeFill'gru/gru_cell_2/ones_like/Shape:output:0'gru/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/ones_like
gru/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell_2/dropout/Const¼
gru/gru_cell_2/dropout/MulMul!gru/gru_cell_2/ones_like:output:0%gru/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout/Mul
gru/gru_cell_2/dropout/ShapeShape!gru/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell_2/dropout/Shapeþ
3gru/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform%gru/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2õ´Ú25
3gru/gru_cell_2/dropout/random_uniform/RandomUniform
%gru/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%gru/gru_cell_2/dropout/GreaterEqual/yû
#gru/gru_cell_2/dropout/GreaterEqualGreaterEqual<gru/gru_cell_2/dropout/random_uniform/RandomUniform:output:0.gru/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#gru/gru_cell_2/dropout/GreaterEqual­
gru/gru_cell_2/dropout/CastCast'gru/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout/Cast·
gru/gru_cell_2/dropout/Mul_1Mulgru/gru_cell_2/dropout/Mul:z:0gru/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout/Mul_1
gru/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru/gru_cell_2/dropout_1/ConstÂ
gru/gru_cell_2/dropout_1/MulMul!gru/gru_cell_2/ones_like:output:0'gru/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout_1/Mul
gru/gru_cell_2/dropout_1/ShapeShape!gru/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/dropout_1/Shape
5gru/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform'gru/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Ô¬27
5gru/gru_cell_2/dropout_1/random_uniform/RandomUniform
'gru/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'gru/gru_cell_2/dropout_1/GreaterEqual/y
%gru/gru_cell_2/dropout_1/GreaterEqualGreaterEqual>gru/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:00gru/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%gru/gru_cell_2/dropout_1/GreaterEqual³
gru/gru_cell_2/dropout_1/CastCast)gru/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout_1/Cast¿
gru/gru_cell_2/dropout_1/Mul_1Mul gru/gru_cell_2/dropout_1/Mul:z:0!gru/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
gru/gru_cell_2/dropout_1/Mul_1
gru/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru/gru_cell_2/dropout_2/ConstÂ
gru/gru_cell_2/dropout_2/MulMul!gru/gru_cell_2/ones_like:output:0'gru/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout_2/Mul
gru/gru_cell_2/dropout_2/ShapeShape!gru/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/dropout_2/Shape
5gru/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform'gru/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Ç27
5gru/gru_cell_2/dropout_2/random_uniform/RandomUniform
'gru/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'gru/gru_cell_2/dropout_2/GreaterEqual/y
%gru/gru_cell_2/dropout_2/GreaterEqualGreaterEqual>gru/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:00gru/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%gru/gru_cell_2/dropout_2/GreaterEqual³
gru/gru_cell_2/dropout_2/CastCast)gru/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/dropout_2/Cast¿
gru/gru_cell_2/dropout_2/Mul_1Mul gru/gru_cell_2/dropout_2/Mul:z:0!gru/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
gru/gru_cell_2/dropout_2/Mul_1
 gru/gru_cell_2/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2"
 gru/gru_cell_2/ones_like_1/Shape
 gru/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 gru/gru_cell_2/ones_like_1/ConstÈ
gru/gru_cell_2/ones_like_1Fill)gru/gru_cell_2/ones_like_1/Shape:output:0)gru/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/ones_like_1
gru/gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru/gru_cell_2/dropout_3/ConstÃ
gru/gru_cell_2/dropout_3/MulMul#gru/gru_cell_2/ones_like_1:output:0'gru/gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/dropout_3/Mul
gru/gru_cell_2/dropout_3/ShapeShape#gru/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/dropout_3/Shape
5gru/gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform'gru/gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ìþi27
5gru/gru_cell_2/dropout_3/random_uniform/RandomUniform
'gru/gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'gru/gru_cell_2/dropout_3/GreaterEqual/y
%gru/gru_cell_2/dropout_3/GreaterEqualGreaterEqual>gru/gru_cell_2/dropout_3/random_uniform/RandomUniform:output:00gru/gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gru/gru_cell_2/dropout_3/GreaterEqual²
gru/gru_cell_2/dropout_3/CastCast)gru/gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/dropout_3/Cast¾
gru/gru_cell_2/dropout_3/Mul_1Mul gru/gru_cell_2/dropout_3/Mul:z:0!gru/gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/gru_cell_2/dropout_3/Mul_1
gru/gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru/gru_cell_2/dropout_4/ConstÃ
gru/gru_cell_2/dropout_4/MulMul#gru/gru_cell_2/ones_like_1:output:0'gru/gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/dropout_4/Mul
gru/gru_cell_2/dropout_4/ShapeShape#gru/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/dropout_4/Shape
5gru/gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform'gru/gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Õê/27
5gru/gru_cell_2/dropout_4/random_uniform/RandomUniform
'gru/gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'gru/gru_cell_2/dropout_4/GreaterEqual/y
%gru/gru_cell_2/dropout_4/GreaterEqualGreaterEqual>gru/gru_cell_2/dropout_4/random_uniform/RandomUniform:output:00gru/gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gru/gru_cell_2/dropout_4/GreaterEqual²
gru/gru_cell_2/dropout_4/CastCast)gru/gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/dropout_4/Cast¾
gru/gru_cell_2/dropout_4/Mul_1Mul gru/gru_cell_2/dropout_4/Mul:z:0!gru/gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/gru_cell_2/dropout_4/Mul_1
gru/gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru/gru_cell_2/dropout_5/ConstÃ
gru/gru_cell_2/dropout_5/MulMul#gru/gru_cell_2/ones_like_1:output:0'gru/gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/dropout_5/Mul
gru/gru_cell_2/dropout_5/ShapeShape#gru/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/dropout_5/Shape
5gru/gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform'gru/gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ý÷27
5gru/gru_cell_2/dropout_5/random_uniform/RandomUniform
'gru/gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'gru/gru_cell_2/dropout_5/GreaterEqual/y
%gru/gru_cell_2/dropout_5/GreaterEqualGreaterEqual>gru/gru_cell_2/dropout_5/random_uniform/RandomUniform:output:00gru/gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gru/gru_cell_2/dropout_5/GreaterEqual²
gru/gru_cell_2/dropout_5/CastCast)gru/gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/dropout_5/Cast¾
gru/gru_cell_2/dropout_5/Mul_1Mul gru/gru_cell_2/dropout_5/Mul:z:0!gru/gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/gru_cell_2/dropout_5/Mul_1¥
gru/gru_cell_2/ReadVariableOpReadVariableOp&gru_gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell_2/ReadVariableOp
gru/gru_cell_2/unstackUnpack%gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell_2/unstack¢
gru/gru_cell_2/mulMulgru/strided_slice_2:output:0 gru/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/mul¨
gru/gru_cell_2/mul_1Mulgru/strided_slice_2:output:0"gru/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/mul_1¨
gru/gru_cell_2/mul_2Mulgru/strided_slice_2:output:0"gru/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/mul_2¬
gru/gru_cell_2/ReadVariableOp_1ReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02!
gru/gru_cell_2/ReadVariableOp_1
"gru/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell_2/strided_slice/stack
$gru/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice/stack_1
$gru/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell_2/strided_slice/stack_2Ù
gru/gru_cell_2/strided_sliceStridedSlice'gru/gru_cell_2/ReadVariableOp_1:value:0+gru/gru_cell_2/strided_slice/stack:output:0-gru/gru_cell_2/strided_slice/stack_1:output:0-gru/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell_2/strided_slice©
gru/gru_cell_2/MatMulMatMulgru/gru_cell_2/mul:z:0%gru/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul¬
gru/gru_cell_2/ReadVariableOp_2ReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02!
gru/gru_cell_2/ReadVariableOp_2
$gru/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice_1/stack¡
&gru/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&gru/gru_cell_2/strided_slice_1/stack_1¡
&gru/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_1/stack_2ã
gru/gru_cell_2/strided_slice_1StridedSlice'gru/gru_cell_2/ReadVariableOp_2:value:0-gru/gru_cell_2/strided_slice_1/stack:output:0/gru/gru_cell_2/strided_slice_1/stack_1:output:0/gru/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_1±
gru/gru_cell_2/MatMul_1MatMulgru/gru_cell_2/mul_1:z:0'gru/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_1¬
gru/gru_cell_2/ReadVariableOp_3ReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02!
gru/gru_cell_2/ReadVariableOp_3
$gru/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell_2/strided_slice_2/stack¡
&gru/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/gru_cell_2/strided_slice_2/stack_1¡
&gru/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_2/stack_2ã
gru/gru_cell_2/strided_slice_2StridedSlice'gru/gru_cell_2/ReadVariableOp_3:value:0-gru/gru_cell_2/strided_slice_2/stack:output:0/gru/gru_cell_2/strided_slice_2/stack_1:output:0/gru/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_2±
gru/gru_cell_2/MatMul_2MatMulgru/gru_cell_2/mul_2:z:0'gru/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_2
$gru/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_3/stack
&gru/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&gru/gru_cell_2/strided_slice_3/stack_1
&gru/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_3/stack_2Æ
gru/gru_cell_2/strided_slice_3StridedSlicegru/gru_cell_2/unstack:output:0-gru/gru_cell_2/strided_slice_3/stack:output:0/gru/gru_cell_2/strided_slice_3/stack_1:output:0/gru/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
gru/gru_cell_2/strided_slice_3·
gru/gru_cell_2/BiasAddBiasAddgru/gru_cell_2/MatMul:product:0'gru/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd
$gru/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_4/stack
&gru/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&gru/gru_cell_2/strided_slice_4/stack_1
&gru/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_4/stack_2´
gru/gru_cell_2/strided_slice_4StridedSlicegru/gru_cell_2/unstack:output:0-gru/gru_cell_2/strided_slice_4/stack:output:0/gru/gru_cell_2/strided_slice_4/stack_1:output:0/gru/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
gru/gru_cell_2/strided_slice_4½
gru/gru_cell_2/BiasAdd_1BiasAdd!gru/gru_cell_2/MatMul_1:product:0'gru/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_1
$gru/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell_2/strided_slice_5/stack
&gru/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&gru/gru_cell_2/strided_slice_5/stack_1
&gru/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_5/stack_2Ä
gru/gru_cell_2/strided_slice_5StridedSlicegru/gru_cell_2/unstack:output:0-gru/gru_cell_2/strided_slice_5/stack:output:0/gru/gru_cell_2/strided_slice_5/stack_1:output:0/gru/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
gru/gru_cell_2/strided_slice_5½
gru/gru_cell_2/BiasAdd_2BiasAdd!gru/gru_cell_2/MatMul_2:product:0'gru/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_2
gru/gru_cell_2/mul_3Mulgru/zeros:output:0"gru/gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_3
gru/gru_cell_2/mul_4Mulgru/zeros:output:0"gru/gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_4
gru/gru_cell_2/mul_5Mulgru/zeros:output:0"gru/gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_5«
gru/gru_cell_2/ReadVariableOp_4ReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02!
gru/gru_cell_2/ReadVariableOp_4
$gru/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice_6/stack¡
&gru/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/gru_cell_2/strided_slice_6/stack_1¡
&gru/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_6/stack_2â
gru/gru_cell_2/strided_slice_6StridedSlice'gru/gru_cell_2/ReadVariableOp_4:value:0-gru/gru_cell_2/strided_slice_6/stack:output:0/gru/gru_cell_2/strided_slice_6/stack_1:output:0/gru/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_6±
gru/gru_cell_2/MatMul_3MatMulgru/gru_cell_2/mul_3:z:0'gru/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_3«
gru/gru_cell_2/ReadVariableOp_5ReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02!
gru/gru_cell_2/ReadVariableOp_5
$gru/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice_7/stack¡
&gru/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&gru/gru_cell_2/strided_slice_7/stack_1¡
&gru/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_7/stack_2â
gru/gru_cell_2/strided_slice_7StridedSlice'gru/gru_cell_2/ReadVariableOp_5:value:0-gru/gru_cell_2/strided_slice_7/stack:output:0/gru/gru_cell_2/strided_slice_7/stack_1:output:0/gru/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_7±
gru/gru_cell_2/MatMul_4MatMulgru/gru_cell_2/mul_4:z:0'gru/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_4
$gru/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_8/stack
&gru/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&gru/gru_cell_2/strided_slice_8/stack_1
&gru/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_8/stack_2Æ
gru/gru_cell_2/strided_slice_8StridedSlicegru/gru_cell_2/unstack:output:1-gru/gru_cell_2/strided_slice_8/stack:output:0/gru/gru_cell_2/strided_slice_8/stack_1:output:0/gru/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
gru/gru_cell_2/strided_slice_8½
gru/gru_cell_2/BiasAdd_3BiasAdd!gru/gru_cell_2/MatMul_3:product:0'gru/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_3
$gru/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_9/stack
&gru/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&gru/gru_cell_2/strided_slice_9/stack_1
&gru/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_9/stack_2´
gru/gru_cell_2/strided_slice_9StridedSlicegru/gru_cell_2/unstack:output:1-gru/gru_cell_2/strided_slice_9/stack:output:0/gru/gru_cell_2/strided_slice_9/stack_1:output:0/gru/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
gru/gru_cell_2/strided_slice_9½
gru/gru_cell_2/BiasAdd_4BiasAdd!gru/gru_cell_2/MatMul_4:product:0'gru/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_4§
gru/gru_cell_2/addAddV2gru/gru_cell_2/BiasAdd:output:0!gru/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add
gru/gru_cell_2/SigmoidSigmoidgru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/Sigmoid­
gru/gru_cell_2/add_1AddV2!gru/gru_cell_2/BiasAdd_1:output:0!gru/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add_1
gru/gru_cell_2/Sigmoid_1Sigmoidgru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/Sigmoid_1«
gru/gru_cell_2/ReadVariableOp_6ReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02!
gru/gru_cell_2/ReadVariableOp_6
%gru/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%gru/gru_cell_2/strided_slice_10/stack£
'gru/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'gru/gru_cell_2/strided_slice_10/stack_1£
'gru/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'gru/gru_cell_2/strided_slice_10/stack_2ç
gru/gru_cell_2/strided_slice_10StridedSlice'gru/gru_cell_2/ReadVariableOp_6:value:0.gru/gru_cell_2/strided_slice_10/stack:output:00gru/gru_cell_2/strided_slice_10/stack_1:output:00gru/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
gru/gru_cell_2/strided_slice_10²
gru/gru_cell_2/MatMul_5MatMulgru/gru_cell_2/mul_5:z:0(gru/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_5
%gru/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%gru/gru_cell_2/strided_slice_11/stack
'gru/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'gru/gru_cell_2/strided_slice_11/stack_1
'gru/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'gru/gru_cell_2/strided_slice_11/stack_2É
gru/gru_cell_2/strided_slice_11StridedSlicegru/gru_cell_2/unstack:output:1.gru/gru_cell_2/strided_slice_11/stack:output:00gru/gru_cell_2/strided_slice_11/stack_1:output:00gru/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
gru/gru_cell_2/strided_slice_11¾
gru/gru_cell_2/BiasAdd_5BiasAdd!gru/gru_cell_2/MatMul_5:product:0(gru/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_5¦
gru/gru_cell_2/mul_6Mulgru/gru_cell_2/Sigmoid_1:y:0!gru/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_6¤
gru/gru_cell_2/add_2AddV2!gru/gru_cell_2/BiasAdd_2:output:0gru/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add_2~
gru/gru_cell_2/TanhTanhgru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/Tanh
gru/gru_cell_2/mul_7Mulgru/gru_cell_2/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_7q
gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell_2/sub/x
gru/gru_cell_2/subSubgru/gru_cell_2/sub/x:output:0gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/sub
gru/gru_cell_2/mul_8Mulgru/gru_cell_2/sub:z:0gru/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_8
gru/gru_cell_2/add_3AddV2gru/gru_cell_2/mul_7:z:0gru/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add_3
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2#
!gru/TensorArrayV2_1/element_shapeÈ
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time
!gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/TensorArrayV2_2/element_shapeÈ
gru/TensorArrayV2_2TensorListReserve*gru/TensorArrayV2_2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
gru/TensorArrayV2_2Ë
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shape
-gru/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorgru/transpose_1:y:0Dgru/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02/
-gru/TensorArrayUnstack_1/TensorListFromTensory
gru/zeros_like	ZerosLikegru/gru_cell_2/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/zeros_like
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counterÐ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0&gru_gru_cell_2_readvariableop_resource(gru_gru_cell_2_readvariableop_1_resource(gru_gru_cell_2_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
gru_while_body_46330* 
condR
gru_while_cond_46329*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : *
parallel_iterations 2
	gru/while½
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    26
4gru/TensorArrayV2Stack/TensorListStack/element_shape
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
gru/strided_slice_3/stack
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2²
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gru/strided_slice_3
gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_2/perm¾
gru/transpose_2	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
gru/transpose_2n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime«
output/Tensordot/ReadVariableOpReadVariableOp(output_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02!
output/Tensordot/ReadVariableOpx
output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/axes
output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
output/Tensordot/frees
output/Tensordot/ShapeShapegru/transpose_2:y:0*
T0*
_output_shapes
:2
output/Tensordot/Shape
output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/GatherV2/axisô
output/Tensordot/GatherV2GatherV2output/Tensordot/Shape:output:0output/Tensordot/free:output:0'output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2
 output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 output/Tensordot/GatherV2_1/axisú
output/Tensordot/GatherV2_1GatherV2output/Tensordot/Shape:output:0output/Tensordot/axes:output:0)output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2_1z
output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const
output/Tensordot/ProdProd"output/Tensordot/GatherV2:output:0output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod~
output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const_1¤
output/Tensordot/Prod_1Prod$output/Tensordot/GatherV2_1:output:0!output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod_1~
output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
output/Tensordot/concat/axisÓ
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat¨
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stack»
output/Tensordot/transpose	Transposegru/transpose_2:y:0 output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
output/Tensordot/transpose»
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
output/Tensordot/Reshapeº
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/Tensordot/MatMul~
output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/Const_2
output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/concat_1/axisà
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1µ
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
output/Tensordot¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp¬
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
output/BiasAddÜ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulï
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mul°
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell_2/ReadVariableOp ^gru/gru_cell_2/ReadVariableOp_1 ^gru/gru_cell_2/ReadVariableOp_2 ^gru/gru_cell_2/ReadVariableOp_3 ^gru/gru_cell_2/ReadVariableOp_4 ^gru/gru_cell_2/ReadVariableOp_5 ^gru/gru_cell_2/ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2>
gru/gru_cell_2/ReadVariableOpgru/gru_cell_2/ReadVariableOp2B
gru/gru_cell_2/ReadVariableOp_1gru/gru_cell_2/ReadVariableOp_12B
gru/gru_cell_2/ReadVariableOp_2gru/gru_cell_2/ReadVariableOp_22B
gru/gru_cell_2/ReadVariableOp_3gru/gru_cell_2/ReadVariableOp_32B
gru/gru_cell_2/ReadVariableOp_4gru/gru_cell_2/ReadVariableOp_42B
gru/gru_cell_2/ReadVariableOp_5gru/gru_cell_2/ReadVariableOp_52B
gru/gru_cell_2/ReadVariableOp_6gru/gru_cell_2/ReadVariableOp_62r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2
	gru/while	gru/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2B
output/Tensordot/ReadVariableOpoutput/Tensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
õ
¥
while_cond_44106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_44106___redundant_placeholder03
/while_while_cond_44106___redundant_placeholder13
/while_while_cond_44106___redundant_placeholder23
/while_while_cond_44106___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ñ
Ú
>__inference_gru_layer_call_and_return_conditional_losses_48027

inputs4
"gru_cell_2_readvariableop_resource:`7
$gru_cell_2_readvariableop_1_resource:	¬`6
$gru_cell_2_readvariableop_4_resource: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell_2/ReadVariableOp¢gru_cell_2/ReadVariableOp_1¢gru_cell_2/ReadVariableOp_2¢gru_cell_2/ReadVariableOp_3¢gru_cell_2/ReadVariableOp_4¢gru_cell_2/ReadVariableOp_5¢gru_cell_2/ReadVariableOp_6¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2
gru_cell_2/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like/Const±
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/ones_likey
gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout/Const¬
gru_cell_2/dropout/MulMulgru_cell_2/ones_like:output:0!gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Mul
gru_cell_2/dropout/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout/Shapeò
/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2õñï21
/gru_cell_2/dropout/random_uniform/RandomUniform
!gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell_2/dropout/GreaterEqual/yë
gru_cell_2/dropout/GreaterEqualGreaterEqual8gru_cell_2/dropout/random_uniform/RandomUniform:output:0*gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell_2/dropout/GreaterEqual¡
gru_cell_2/dropout/CastCast#gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Cast§
gru_cell_2/dropout/Mul_1Mulgru_cell_2/dropout/Mul:z:0gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Mul_1}
gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_1/Const²
gru_cell_2/dropout_1/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Mul
gru_cell_2/dropout_1/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_1/Shapeø
1gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ø­ê23
1gru_cell_2/dropout_1/random_uniform/RandomUniform
#gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_1/GreaterEqual/yó
!gru_cell_2/dropout_1/GreaterEqualGreaterEqual:gru_cell_2/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru_cell_2/dropout_1/GreaterEqual§
gru_cell_2/dropout_1/CastCast%gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Cast¯
gru_cell_2/dropout_1/Mul_1Mulgru_cell_2/dropout_1/Mul:z:0gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Mul_1}
gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_2/Const²
gru_cell_2/dropout_2/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Mul
gru_cell_2/dropout_2/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_2/Shapeø
1gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ñº¤23
1gru_cell_2/dropout_2/random_uniform/RandomUniform
#gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_2/GreaterEqual/yó
!gru_cell_2/dropout_2/GreaterEqualGreaterEqual:gru_cell_2/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru_cell_2/dropout_2/GreaterEqual§
gru_cell_2/dropout_2/CastCast%gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Cast¯
gru_cell_2/dropout_2/Mul_1Mulgru_cell_2/dropout_2/Mul:z:0gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Mul_1z
gru_cell_2/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like_1/Shape
gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like_1/Const¸
gru_cell_2/ones_like_1Fill%gru_cell_2/ones_like_1/Shape:output:0%gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/ones_like_1}
gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_3/Const³
gru_cell_2/dropout_3/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Mul
gru_cell_2/dropout_3/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_3/Shape÷
1gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ü¦ 23
1gru_cell_2/dropout_3/random_uniform/RandomUniform
#gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_3/GreaterEqual/yò
!gru_cell_2/dropout_3/GreaterEqualGreaterEqual:gru_cell_2/dropout_3/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_3/GreaterEqual¦
gru_cell_2/dropout_3/CastCast%gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Cast®
gru_cell_2/dropout_3/Mul_1Mulgru_cell_2/dropout_3/Mul:z:0gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Mul_1}
gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_4/Const³
gru_cell_2/dropout_4/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Mul
gru_cell_2/dropout_4/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_4/Shape÷
1gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2£ô23
1gru_cell_2/dropout_4/random_uniform/RandomUniform
#gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_4/GreaterEqual/yò
!gru_cell_2/dropout_4/GreaterEqualGreaterEqual:gru_cell_2/dropout_4/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_4/GreaterEqual¦
gru_cell_2/dropout_4/CastCast%gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Cast®
gru_cell_2/dropout_4/Mul_1Mulgru_cell_2/dropout_4/Mul:z:0gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Mul_1}
gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_5/Const³
gru_cell_2/dropout_5/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Mul
gru_cell_2/dropout_5/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_5/Shape÷
1gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ÈÙ¼23
1gru_cell_2/dropout_5/random_uniform/RandomUniform
#gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_5/GreaterEqual/yò
!gru_cell_2/dropout_5/GreaterEqualGreaterEqual:gru_cell_2/dropout_5/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_5/GreaterEqual¦
gru_cell_2/dropout_5/CastCast%gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Cast®
gru_cell_2/dropout_5/Mul_1Mulgru_cell_2/dropout_5/Mul:z:0gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Mul_1
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_2/ReadVariableOp
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_2/unstack
gru_cell_2/mulMulstrided_slice_2:output:0gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul
gru_cell_2/mul_1Mulstrided_slice_2:output:0gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_1
gru_cell_2/mul_2Mulstrided_slice_2:output:0gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_2 
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_1
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice/stack_1
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2Á
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice
gru_cell_2/MatMulMatMulgru_cell_2/mul:z:0!gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul 
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_2
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_1/stack
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_1/stack_1
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2Ë
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_1¡
gru_cell_2/MatMul_1MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_1 
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_3
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell_2/strided_slice_2/stack
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2Ë
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_2¡
gru_cell_2/MatMul_2MatMulgru_cell_2/mul_2:z:0#gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_2
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_3/stack_1
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2®
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_3§
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_4/stack
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_4/stack_1
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_4­
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_1
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell_2/strided_slice_5/stack
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2¬
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_5­
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_2
gru_cell_2/mul_3Mulzeros:output:0gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_3
gru_cell_2/mul_4Mulzeros:output:0gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_4
gru_cell_2/mul_5Mulzeros:output:0gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_5
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_4
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_6/stack_1
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2Ê
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_6¡
gru_cell_2/MatMul_3MatMulgru_cell_2/mul_3:z:0#gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_3
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_5
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_7/stack
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_7/stack_1
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2Ê
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_7¡
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_4:z:0#gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_4
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_8/stack_1
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2®
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_8­
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_3
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_9/stack
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_9/stack_1
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_9­
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_4
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/addy
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_1
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid_1
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_6
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!gru_cell_2/strided_slice_10/stack
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2Ï
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_10¢
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_5:z:0$gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_5
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2#
!gru_cell_2/strided_slice_11/stack
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2±
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_11®
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_5
gru_cell_2/mul_6Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_6
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_2r
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Tanh
gru_cell_2/mul_7Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_7i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/sub/x
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/sub
gru_cell_2/mul_8Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_8
gru_cell_2/add_3AddV2gru_cell_2/mul_7:z:0gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_47815*
condR
while_cond_47814*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulë
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÆ
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ô
Ú
>__inference_gru_layer_call_and_return_conditional_losses_45087

inputs4
"gru_cell_2_readvariableop_resource:`7
$gru_cell_2_readvariableop_1_resource:	¬`6
$gru_cell_2_readvariableop_4_resource: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell_2/ReadVariableOp¢gru_cell_2/ReadVariableOp_1¢gru_cell_2/ReadVariableOp_2¢gru_cell_2/ReadVariableOp_3¢gru_cell_2/ReadVariableOp_4¢gru_cell_2/ReadVariableOp_5¢gru_cell_2/ReadVariableOp_6¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2
gru_cell_2/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like/Const±
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/ones_likez
gru_cell_2/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like_1/Shape
gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like_1/Const¸
gru_cell_2/ones_like_1Fill%gru_cell_2/ones_like_1/Shape:output:0%gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/ones_like_1
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_2/ReadVariableOp
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_2/unstack
gru_cell_2/mulMulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul
gru_cell_2/mul_1Mulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_1
gru_cell_2/mul_2Mulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_2 
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_1
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice/stack_1
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2Á
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice
gru_cell_2/MatMulMatMulgru_cell_2/mul:z:0!gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul 
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_2
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_1/stack
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_1/stack_1
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2Ë
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_1¡
gru_cell_2/MatMul_1MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_1 
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_3
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell_2/strided_slice_2/stack
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2Ë
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_2¡
gru_cell_2/MatMul_2MatMulgru_cell_2/mul_2:z:0#gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_2
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_3/stack_1
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2®
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_3§
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_4/stack
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_4/stack_1
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_4­
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_1
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell_2/strided_slice_5/stack
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2¬
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_5­
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_2
gru_cell_2/mul_3Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_3
gru_cell_2/mul_4Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_4
gru_cell_2/mul_5Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_5
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_4
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_6/stack_1
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2Ê
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_6¡
gru_cell_2/MatMul_3MatMulgru_cell_2/mul_3:z:0#gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_3
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_5
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_7/stack
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_7/stack_1
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2Ê
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_7¡
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_4:z:0#gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_4
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_8/stack_1
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2®
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_8­
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_3
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_9/stack
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_9/stack_1
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_9­
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_4
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/addy
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_1
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid_1
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_6
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!gru_cell_2/strided_slice_10/stack
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2Ï
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_10¢
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_5:z:0$gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_5
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2#
!gru_cell_2/strided_slice_11/stack
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2±
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_11®
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_5
gru_cell_2/mul_6Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_6
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_2r
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Tanh
gru_cell_2/mul_7Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_7i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/sub/x
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/sub
gru_cell_2/mul_8Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_8
gru_cell_2/add_3AddV2gru_cell_2/mul_7:z:0gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44923*
condR
while_cond_44922*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulë
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÆ
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
õ
¥
while_cond_44922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_44922___redundant_placeholder03
/while_while_cond_44922___redundant_placeholder13
/while_while_cond_44922___redundant_placeholder23
/while_while_cond_44922___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
â
ò
.__inference_GRU_classifier_layer_call_fn_45157	
input
unknown:`
	unknown_0:	¬`
	unknown_1: `
	unknown_2: 
	unknown_3:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8 *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_451442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
ã'
¹
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45144

inputs
	gru_45088:`
	gru_45090:	¬`
	gru_45092: `
output_45126: 
output_45128:
identity¢gru/StatefulPartitionedCall¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallã
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_447912
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_45088	gru_45090	gru_45092*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_450872
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_45126output_45128*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_451252 
output/StatefulPartitionedCall½
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45090*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulÐ
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45092*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall8^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
å
ó
.__inference_GRU_classifier_layer_call_fn_45768

inputs
unknown:`
	unknown_0:	¬`
	unknown_1: `
	unknown_2: 
	unknown_3:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8 *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_451442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
õ
¥
while_cond_45358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45358___redundant_placeholder03
/while_while_cond_45358___redundant_placeholder13
/while_while_cond_45358___redundant_placeholder23
/while_while_cond_45358___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
â
Ð
__inference_loss_fn_1_48404\
Jgru_gru_cell_2_recurrent_kernel_regularizer_square_readvariableop_resource: `
identity¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpJgru_gru_cell_2_recurrent_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulº
IdentityIdentity3gru/gru_cell_2/recurrent_kernel/Regularizer/mul:z:0B^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp
õ
¥
while_cond_46785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46785___redundant_placeholder03
/while_while_cond_46785___redundant_placeholder13
/while_while_cond_46785___redundant_placeholder23
/while_while_cond_46785___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ÿ·
æ
while_body_44923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`?
,while_gru_cell_2_readvariableop_1_resource_0:	¬`>
,while_gru_cell_2_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`=
*while_gru_cell_2_readvariableop_1_resource:	¬`<
*while_gru_cell_2_readvariableop_4_resource: `¢while/gru_cell_2/ReadVariableOp¢!while/gru_cell_2/ReadVariableOp_1¢!while/gru_cell_2/ReadVariableOp_2¢!while/gru_cell_2/ReadVariableOp_3¢!while/gru_cell_2/ReadVariableOp_4¢!while/gru_cell_2/ReadVariableOp_5¢!while/gru_cell_2/ReadVariableOp_6Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¤
 while/gru_cell_2/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell_2/ones_like/ConstÉ
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/ones_like
"while/gru_cell_2/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/gru_cell_2/ones_like_1/Shape
"while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/gru_cell_2/ones_like_1/ConstÐ
while/gru_cell_2/ones_like_1Fill+while/gru_cell_2/ones_like_1/Shape:output:0+while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/ones_like_1­
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02!
while/gru_cell_2/ReadVariableOp
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell_2/unstack½
while/gru_cell_2/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mulÁ
while/gru_cell_2/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_1Á
while/gru_cell_2/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_2´
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_1
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack¡
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice/stack_1¡
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2å
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice±
while/gru_cell_2/MatMulMatMulwhile/gru_cell_2/mul:z:0'while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul´
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_2¡
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_1/stack¥
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_1/stack_1¥
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2ï
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1¹
while/gru_cell_2/MatMul_1MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_1´
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_3¡
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell_2/strided_slice_2/stack¥
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1¥
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2ï
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2¹
while/gru_cell_2/MatMul_2MatMulwhile/gru_cell_2/mul_2:z:0)while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_2
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_3/stack_1
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2Ò
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_3¿
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_4/stack
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_4/stack_1
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2À
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_4Å
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_1
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell_2/strided_slice_5/stack
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2Ð
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2"
 while/gru_cell_2/strided_slice_5Å
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_2¥
while/gru_cell_2/mul_3Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_3¥
while/gru_cell_2/mul_4Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_4¥
while/gru_cell_2/mul_5Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_5³
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_4¡
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack¥
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_6/stack_1¥
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2î
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6¹
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul_3:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_3³
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_5¡
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_7/stack¥
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_7/stack_1¥
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2î
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7¹
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_4:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_4
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_8/stack_1
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2Ò
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_8Å
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_3
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_9/stack
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_9/stack_1
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2À
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_9Å
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_4¯
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoidµ
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_1
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoid_1³
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_6£
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/gru_cell_2/strided_slice_10/stack§
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1§
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2ó
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10º
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_5:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_5
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2)
'while/gru_cell_2/strided_slice_11/stack 
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1 
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2Õ
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2#
!while/gru_cell_2/strided_slice_11Æ
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_5®
while/gru_cell_2/mul_6Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_6¬
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_2
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Tanh
while/gru_cell_2/mul_7Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_7u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_2/sub/x¤
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/sub
while/gru_cell_2/mul_8Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_8£
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_7:z:0while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ø
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityë
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ú
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ø
Ü
>__inference_gru_layer_call_and_return_conditional_losses_47341
inputs_04
"gru_cell_2_readvariableop_resource:`7
$gru_cell_2_readvariableop_1_resource:	¬`6
$gru_cell_2_readvariableop_4_resource: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell_2/ReadVariableOp¢gru_cell_2/ReadVariableOp_1¢gru_cell_2/ReadVariableOp_2¢gru_cell_2/ReadVariableOp_3¢gru_cell_2/ReadVariableOp_4¢gru_cell_2/ReadVariableOp_5¢gru_cell_2/ReadVariableOp_6¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2
gru_cell_2/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like/Const±
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/ones_likey
gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout/Const¬
gru_cell_2/dropout/MulMulgru_cell_2/ones_like:output:0!gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Mul
gru_cell_2/dropout/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout/Shapeò
/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2óÝ21
/gru_cell_2/dropout/random_uniform/RandomUniform
!gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell_2/dropout/GreaterEqual/yë
gru_cell_2/dropout/GreaterEqualGreaterEqual8gru_cell_2/dropout/random_uniform/RandomUniform:output:0*gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell_2/dropout/GreaterEqual¡
gru_cell_2/dropout/CastCast#gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Cast§
gru_cell_2/dropout/Mul_1Mulgru_cell_2/dropout/Mul:z:0gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout/Mul_1}
gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_1/Const²
gru_cell_2/dropout_1/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Mul
gru_cell_2/dropout_1/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_1/Shapeø
1gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÊÞý23
1gru_cell_2/dropout_1/random_uniform/RandomUniform
#gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_1/GreaterEqual/yó
!gru_cell_2/dropout_1/GreaterEqualGreaterEqual:gru_cell_2/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru_cell_2/dropout_1/GreaterEqual§
gru_cell_2/dropout_1/CastCast%gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Cast¯
gru_cell_2/dropout_1/Mul_1Mulgru_cell_2/dropout_1/Mul:z:0gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_1/Mul_1}
gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_2/Const²
gru_cell_2/dropout_2/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Mul
gru_cell_2/dropout_2/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_2/Shape÷
1gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2#23
1gru_cell_2/dropout_2/random_uniform/RandomUniform
#gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_2/GreaterEqual/yó
!gru_cell_2/dropout_2/GreaterEqualGreaterEqual:gru_cell_2/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru_cell_2/dropout_2/GreaterEqual§
gru_cell_2/dropout_2/CastCast%gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Cast¯
gru_cell_2/dropout_2/Mul_1Mulgru_cell_2/dropout_2/Mul:z:0gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/dropout_2/Mul_1z
gru_cell_2/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like_1/Shape
gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like_1/Const¸
gru_cell_2/ones_like_1Fill%gru_cell_2/ones_like_1/Shape:output:0%gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/ones_like_1}
gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_3/Const³
gru_cell_2/dropout_3/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Mul
gru_cell_2/dropout_3/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_3/Shape÷
1gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2 ¡23
1gru_cell_2/dropout_3/random_uniform/RandomUniform
#gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_3/GreaterEqual/yò
!gru_cell_2/dropout_3/GreaterEqualGreaterEqual:gru_cell_2/dropout_3/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_3/GreaterEqual¦
gru_cell_2/dropout_3/CastCast%gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Cast®
gru_cell_2/dropout_3/Mul_1Mulgru_cell_2/dropout_3/Mul:z:0gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_3/Mul_1}
gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_4/Const³
gru_cell_2/dropout_4/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Mul
gru_cell_2/dropout_4/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_4/Shape÷
1gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2´ç¥23
1gru_cell_2/dropout_4/random_uniform/RandomUniform
#gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_4/GreaterEqual/yò
!gru_cell_2/dropout_4/GreaterEqualGreaterEqual:gru_cell_2/dropout_4/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_4/GreaterEqual¦
gru_cell_2/dropout_4/CastCast%gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Cast®
gru_cell_2/dropout_4/Mul_1Mulgru_cell_2/dropout_4/Mul:z:0gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_4/Mul_1}
gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_5/Const³
gru_cell_2/dropout_5/MulMulgru_cell_2/ones_like_1:output:0#gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Mul
gru_cell_2/dropout_5/ShapeShapegru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_5/Shape÷
1gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2¹Ý23
1gru_cell_2/dropout_5/random_uniform/RandomUniform
#gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru_cell_2/dropout_5/GreaterEqual/yò
!gru_cell_2/dropout_5/GreaterEqualGreaterEqual:gru_cell_2/dropout_5/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru_cell_2/dropout_5/GreaterEqual¦
gru_cell_2/dropout_5/CastCast%gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Cast®
gru_cell_2/dropout_5/Mul_1Mulgru_cell_2/dropout_5/Mul:z:0gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/dropout_5/Mul_1
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_2/ReadVariableOp
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_2/unstack
gru_cell_2/mulMulstrided_slice_2:output:0gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul
gru_cell_2/mul_1Mulstrided_slice_2:output:0gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_1
gru_cell_2/mul_2Mulstrided_slice_2:output:0gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_2 
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_1
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice/stack_1
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2Á
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice
gru_cell_2/MatMulMatMulgru_cell_2/mul:z:0!gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul 
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_2
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_1/stack
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_1/stack_1
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2Ë
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_1¡
gru_cell_2/MatMul_1MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_1 
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_3
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell_2/strided_slice_2/stack
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2Ë
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_2¡
gru_cell_2/MatMul_2MatMulgru_cell_2/mul_2:z:0#gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_2
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_3/stack_1
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2®
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_3§
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_4/stack
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_4/stack_1
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_4­
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_1
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell_2/strided_slice_5/stack
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2¬
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_5­
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_2
gru_cell_2/mul_3Mulzeros:output:0gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_3
gru_cell_2/mul_4Mulzeros:output:0gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_4
gru_cell_2/mul_5Mulzeros:output:0gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_5
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_4
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_6/stack_1
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2Ê
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_6¡
gru_cell_2/MatMul_3MatMulgru_cell_2/mul_3:z:0#gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_3
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_5
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_7/stack
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_7/stack_1
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2Ê
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_7¡
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_4:z:0#gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_4
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_8/stack_1
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2®
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_8­
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_3
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_9/stack
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_9/stack_1
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_9­
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_4
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/addy
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_1
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid_1
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_6
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!gru_cell_2/strided_slice_10/stack
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2Ï
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_10¢
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_5:z:0$gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_5
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2#
!gru_cell_2/strided_slice_11/stack
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2±
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_11®
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_5
gru_cell_2/mul_6Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_6
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_2r
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Tanh
gru_cell_2/mul_7Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_7i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/sub/x
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/sub
gru_cell_2/mul_8Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_8
gru_cell_2/add_3AddV2gru_cell_2/mul_7:z:0gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_47129*
condR
while_cond_47128*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulë
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÆ
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
â
ò
.__inference_GRU_classifier_layer_call_fn_45660	
input
unknown:`
	unknown_0:	¬`
	unknown_1: `
	unknown_2: 
	unknown_3:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8 *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_456322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
à'
¸
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45718	
input
	gru_45693:`
	gru_45695:	¬`
	gru_45697: `
output_45700: 
output_45702:
identity¢gru/StatefulPartitionedCall¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallâ
masking/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_447912
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_45693	gru_45695	gru_45697*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_455712
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_45700output_45702*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_451252 
output/StatefulPartitionedCall½
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45695*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulÐ
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45697*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall8^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
ÿ·
æ
while_body_47472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`?
,while_gru_cell_2_readvariableop_1_resource_0:	¬`>
,while_gru_cell_2_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`=
*while_gru_cell_2_readvariableop_1_resource:	¬`<
*while_gru_cell_2_readvariableop_4_resource: `¢while/gru_cell_2/ReadVariableOp¢!while/gru_cell_2/ReadVariableOp_1¢!while/gru_cell_2/ReadVariableOp_2¢!while/gru_cell_2/ReadVariableOp_3¢!while/gru_cell_2/ReadVariableOp_4¢!while/gru_cell_2/ReadVariableOp_5¢!while/gru_cell_2/ReadVariableOp_6Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¤
 while/gru_cell_2/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell_2/ones_like/ConstÉ
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/ones_like
"while/gru_cell_2/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/gru_cell_2/ones_like_1/Shape
"while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/gru_cell_2/ones_like_1/ConstÐ
while/gru_cell_2/ones_like_1Fill+while/gru_cell_2/ones_like_1/Shape:output:0+while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/ones_like_1­
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02!
while/gru_cell_2/ReadVariableOp
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell_2/unstack½
while/gru_cell_2/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mulÁ
while/gru_cell_2/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_1Á
while/gru_cell_2/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_2´
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_1
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack¡
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice/stack_1¡
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2å
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice±
while/gru_cell_2/MatMulMatMulwhile/gru_cell_2/mul:z:0'while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul´
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_2¡
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_1/stack¥
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_1/stack_1¥
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2ï
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1¹
while/gru_cell_2/MatMul_1MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_1´
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_3¡
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell_2/strided_slice_2/stack¥
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1¥
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2ï
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2¹
while/gru_cell_2/MatMul_2MatMulwhile/gru_cell_2/mul_2:z:0)while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_2
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_3/stack_1
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2Ò
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_3¿
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_4/stack
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_4/stack_1
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2À
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_4Å
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_1
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell_2/strided_slice_5/stack
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2Ð
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2"
 while/gru_cell_2/strided_slice_5Å
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_2¥
while/gru_cell_2/mul_3Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_3¥
while/gru_cell_2/mul_4Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_4¥
while/gru_cell_2/mul_5Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_5³
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_4¡
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack¥
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_6/stack_1¥
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2î
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6¹
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul_3:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_3³
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_5¡
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_7/stack¥
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_7/stack_1¥
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2î
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7¹
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_4:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_4
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_8/stack_1
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2Ò
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_8Å
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_3
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_9/stack
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_9/stack_1
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2À
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_9Å
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_4¯
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoidµ
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_1
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoid_1³
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_6£
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/gru_cell_2/strided_slice_10/stack§
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1§
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2ó
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10º
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_5:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_5
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2)
'while/gru_cell_2/strided_slice_11/stack 
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1 
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2Õ
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2#
!while/gru_cell_2/strided_slice_11Æ
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_5®
while/gru_cell_2/mul_6Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_6¬
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_2
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Tanh
while/gru_cell_2/mul_7Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_7u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_2/sub/x¤
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/sub
while/gru_cell_2/mul_8Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_8£
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_7:z:0while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ø
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityë
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ú
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Î

&__inference_output_layer_call_fn_48036

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_451252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
öh
å
!__inference__traced_restore_48581
file_prefix0
assignvariableop_output_kernel: ,
assignvariableop_1_output_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ;
(assignvariableop_7_gru_gru_cell_2_kernel:	¬`D
2assignvariableop_8_gru_gru_cell_2_recurrent_kernel: `8
&assignvariableop_9_gru_gru_cell_2_bias:`#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: :
(assignvariableop_14_adam_output_kernel_m: 4
&assignvariableop_15_adam_output_bias_m:C
0assignvariableop_16_adam_gru_gru_cell_2_kernel_m:	¬`L
:assignvariableop_17_adam_gru_gru_cell_2_recurrent_kernel_m: `@
.assignvariableop_18_adam_gru_gru_cell_2_bias_m:`:
(assignvariableop_19_adam_output_kernel_v: 4
&assignvariableop_20_adam_output_bias_v:C
0assignvariableop_21_adam_gru_gru_cell_2_kernel_v:	¬`L
:assignvariableop_22_adam_gru_gru_cell_2_recurrent_kernel_v: `@
.assignvariableop_23_adam_gru_gru_cell_2_bias_v:`
identity_25¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¨
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_output_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_output_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7­
AssignVariableOp_7AssignVariableOp(assignvariableop_7_gru_gru_cell_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8·
AssignVariableOp_8AssignVariableOp2assignvariableop_8_gru_gru_cell_2_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9«
AssignVariableOp_9AssignVariableOp&assignvariableop_9_gru_gru_cell_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14°
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_output_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_output_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¸
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_gru_gru_cell_2_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Â
AssignVariableOp_17AssignVariableOp:assignvariableop_17_adam_gru_gru_cell_2_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¶
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_gru_gru_cell_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19°
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_output_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_output_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¸
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_gru_gru_cell_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Â
AssignVariableOp_22AssignVariableOp:assignvariableop_22_adam_gru_gru_cell_2_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¶
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_gru_gru_cell_2_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpî
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24á
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ÏÚ
í
gru_while_body_45930$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0c
_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0@
.gru_while_gru_cell_2_readvariableop_resource_0:`C
0gru_while_gru_cell_2_readvariableop_1_resource_0:	¬`B
0gru_while_gru_cell_2_readvariableop_4_resource_0: `
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_identity_5!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensora
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor>
,gru_while_gru_cell_2_readvariableop_resource:`A
.gru_while_gru_cell_2_readvariableop_1_resource:	¬`@
.gru_while_gru_cell_2_readvariableop_4_resource: `¢#gru/while/gru_cell_2/ReadVariableOp¢%gru/while/gru_cell_2/ReadVariableOp_1¢%gru/while/gru_cell_2/ReadVariableOp_2¢%gru/while/gru_cell_2/ReadVariableOp_3¢%gru/while/gru_cell_2/ReadVariableOp_4¢%gru/while/gru_cell_2/ReadVariableOp_5¢%gru/while/gru_cell_2/ReadVariableOp_6Ë
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeì
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItemÏ
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeõ
/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0gru_while_placeholderFgru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
21
/gru/while/TensorArrayV2Read_1/TensorListGetItem°
$gru/while/gru_cell_2/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/ones_like/Shape
$gru/while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$gru/while/gru_cell_2/ones_like/ConstÙ
gru/while/gru_cell_2/ones_likeFill-gru/while/gru_cell_2/ones_like/Shape:output:0-gru/while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
gru/while/gru_cell_2/ones_like
&gru/while/gru_cell_2/ones_like_1/ShapeShapegru_while_placeholder_3*
T0*
_output_shapes
:2(
&gru/while/gru_cell_2/ones_like_1/Shape
&gru/while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&gru/while/gru_cell_2/ones_like_1/Constà
 gru/while/gru_cell_2/ones_like_1Fill/gru/while/gru_cell_2/ones_like_1/Shape:output:0/gru/while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gru/while/gru_cell_2/ones_like_1¹
#gru/while/gru_cell_2/ReadVariableOpReadVariableOp.gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02%
#gru/while/gru_cell_2/ReadVariableOp©
gru/while/gru_cell_2/unstackUnpack+gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell_2/unstackÍ
gru/while/gru_cell_2/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0'gru/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell_2/mulÑ
gru/while/gru_cell_2/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0'gru/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell_2/mul_1Ñ
gru/while/gru_cell_2/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0'gru/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell_2/mul_2À
%gru/while/gru_cell_2/ReadVariableOp_1ReadVariableOp0gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_1¥
(gru/while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell_2/strided_slice/stack©
*gru/while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice/stack_1©
*gru/while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell_2/strided_slice/stack_2ý
"gru/while/gru_cell_2/strided_sliceStridedSlice-gru/while/gru_cell_2/ReadVariableOp_1:value:01gru/while/gru_cell_2/strided_slice/stack:output:03gru/while/gru_cell_2/strided_slice/stack_1:output:03gru/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2$
"gru/while/gru_cell_2/strided_sliceÁ
gru/while/gru_cell_2/MatMulMatMulgru/while/gru_cell_2/mul:z:0+gru/while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMulÀ
%gru/while/gru_cell_2/ReadVariableOp_2ReadVariableOp0gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_2©
*gru/while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice_1/stack­
,gru/while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,gru/while/gru_cell_2/strided_slice_1/stack_1­
,gru/while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_1/stack_2
$gru/while/gru_cell_2/strided_slice_1StridedSlice-gru/while/gru_cell_2/ReadVariableOp_2:value:03gru/while/gru_cell_2/strided_slice_1/stack:output:05gru/while/gru_cell_2/strided_slice_1/stack_1:output:05gru/while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_1É
gru/while/gru_cell_2/MatMul_1MatMulgru/while/gru_cell_2/mul_1:z:0-gru/while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_1À
%gru/while/gru_cell_2/ReadVariableOp_3ReadVariableOp0gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_3©
*gru/while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell_2/strided_slice_2/stack­
,gru/while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,gru/while/gru_cell_2/strided_slice_2/stack_1­
,gru/while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_2/stack_2
$gru/while/gru_cell_2/strided_slice_2StridedSlice-gru/while/gru_cell_2/ReadVariableOp_3:value:03gru/while/gru_cell_2/strided_slice_2/stack:output:05gru/while/gru_cell_2/strided_slice_2/stack_1:output:05gru/while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_2É
gru/while/gru_cell_2/MatMul_2MatMulgru/while/gru_cell_2/mul_2:z:0-gru/while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_2¢
*gru/while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_3/stack¦
,gru/while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,gru/while/gru_cell_2/strided_slice_3/stack_1¦
,gru/while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_3/stack_2ê
$gru/while/gru_cell_2/strided_slice_3StridedSlice%gru/while/gru_cell_2/unstack:output:03gru/while/gru_cell_2/strided_slice_3/stack:output:05gru/while/gru_cell_2/strided_slice_3/stack_1:output:05gru/while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2&
$gru/while/gru_cell_2/strided_slice_3Ï
gru/while/gru_cell_2/BiasAddBiasAdd%gru/while/gru_cell_2/MatMul:product:0-gru/while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/BiasAdd¢
*gru/while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_4/stack¦
,gru/while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2.
,gru/while/gru_cell_2/strided_slice_4/stack_1¦
,gru/while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_4/stack_2Ø
$gru/while/gru_cell_2/strided_slice_4StridedSlice%gru/while/gru_cell_2/unstack:output:03gru/while/gru_cell_2/strided_slice_4/stack:output:05gru/while/gru_cell_2/strided_slice_4/stack_1:output:05gru/while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2&
$gru/while/gru_cell_2/strided_slice_4Õ
gru/while/gru_cell_2/BiasAdd_1BiasAdd'gru/while/gru_cell_2/MatMul_1:product:0-gru/while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_1¢
*gru/while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell_2/strided_slice_5/stack¦
,gru/while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,gru/while/gru_cell_2/strided_slice_5/stack_1¦
,gru/while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_5/stack_2è
$gru/while/gru_cell_2/strided_slice_5StridedSlice%gru/while/gru_cell_2/unstack:output:03gru/while/gru_cell_2/strided_slice_5/stack:output:05gru/while/gru_cell_2/strided_slice_5/stack_1:output:05gru/while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2&
$gru/while/gru_cell_2/strided_slice_5Õ
gru/while/gru_cell_2/BiasAdd_2BiasAdd'gru/while/gru_cell_2/MatMul_2:product:0-gru/while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_2µ
gru/while/gru_cell_2/mul_3Mulgru_while_placeholder_3)gru/while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_3µ
gru/while/gru_cell_2/mul_4Mulgru_while_placeholder_3)gru/while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_4µ
gru/while/gru_cell_2/mul_5Mulgru_while_placeholder_3)gru/while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_5¿
%gru/while/gru_cell_2/ReadVariableOp_4ReadVariableOp0gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_4©
*gru/while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice_6/stack­
,gru/while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,gru/while/gru_cell_2/strided_slice_6/stack_1­
,gru/while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_6/stack_2
$gru/while/gru_cell_2/strided_slice_6StridedSlice-gru/while/gru_cell_2/ReadVariableOp_4:value:03gru/while/gru_cell_2/strided_slice_6/stack:output:05gru/while/gru_cell_2/strided_slice_6/stack_1:output:05gru/while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_6É
gru/while/gru_cell_2/MatMul_3MatMulgru/while/gru_cell_2/mul_3:z:0-gru/while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_3¿
%gru/while/gru_cell_2/ReadVariableOp_5ReadVariableOp0gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_5©
*gru/while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice_7/stack­
,gru/while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,gru/while/gru_cell_2/strided_slice_7/stack_1­
,gru/while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_7/stack_2
$gru/while/gru_cell_2/strided_slice_7StridedSlice-gru/while/gru_cell_2/ReadVariableOp_5:value:03gru/while/gru_cell_2/strided_slice_7/stack:output:05gru/while/gru_cell_2/strided_slice_7/stack_1:output:05gru/while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_7É
gru/while/gru_cell_2/MatMul_4MatMulgru/while/gru_cell_2/mul_4:z:0-gru/while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_4¢
*gru/while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_8/stack¦
,gru/while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,gru/while/gru_cell_2/strided_slice_8/stack_1¦
,gru/while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_8/stack_2ê
$gru/while/gru_cell_2/strided_slice_8StridedSlice%gru/while/gru_cell_2/unstack:output:13gru/while/gru_cell_2/strided_slice_8/stack:output:05gru/while/gru_cell_2/strided_slice_8/stack_1:output:05gru/while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2&
$gru/while/gru_cell_2/strided_slice_8Õ
gru/while/gru_cell_2/BiasAdd_3BiasAdd'gru/while/gru_cell_2/MatMul_3:product:0-gru/while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_3¢
*gru/while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_9/stack¦
,gru/while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2.
,gru/while/gru_cell_2/strided_slice_9/stack_1¦
,gru/while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_9/stack_2Ø
$gru/while/gru_cell_2/strided_slice_9StridedSlice%gru/while/gru_cell_2/unstack:output:13gru/while/gru_cell_2/strided_slice_9/stack:output:05gru/while/gru_cell_2/strided_slice_9/stack_1:output:05gru/while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2&
$gru/while/gru_cell_2/strided_slice_9Õ
gru/while/gru_cell_2/BiasAdd_4BiasAdd'gru/while/gru_cell_2/MatMul_4:product:0-gru/while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_4¿
gru/while/gru_cell_2/addAddV2%gru/while/gru_cell_2/BiasAdd:output:0'gru/while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add
gru/while/gru_cell_2/SigmoidSigmoidgru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/SigmoidÅ
gru/while/gru_cell_2/add_1AddV2'gru/while/gru_cell_2/BiasAdd_1:output:0'gru/while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add_1
gru/while/gru_cell_2/Sigmoid_1Sigmoidgru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/Sigmoid_1¿
%gru/while/gru_cell_2/ReadVariableOp_6ReadVariableOp0gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_6«
+gru/while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2-
+gru/while/gru_cell_2/strided_slice_10/stack¯
-gru/while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-gru/while/gru_cell_2/strided_slice_10/stack_1¯
-gru/while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-gru/while/gru_cell_2/strided_slice_10/stack_2
%gru/while/gru_cell_2/strided_slice_10StridedSlice-gru/while/gru_cell_2/ReadVariableOp_6:value:04gru/while/gru_cell_2/strided_slice_10/stack:output:06gru/while/gru_cell_2/strided_slice_10/stack_1:output:06gru/while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2'
%gru/while/gru_cell_2/strided_slice_10Ê
gru/while/gru_cell_2/MatMul_5MatMulgru/while/gru_cell_2/mul_5:z:0.gru/while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_5¤
+gru/while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2-
+gru/while/gru_cell_2/strided_slice_11/stack¨
-gru/while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-gru/while/gru_cell_2/strided_slice_11/stack_1¨
-gru/while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gru/while/gru_cell_2/strided_slice_11/stack_2í
%gru/while/gru_cell_2/strided_slice_11StridedSlice%gru/while/gru_cell_2/unstack:output:14gru/while/gru_cell_2/strided_slice_11/stack:output:06gru/while/gru_cell_2/strided_slice_11/stack_1:output:06gru/while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2'
%gru/while/gru_cell_2/strided_slice_11Ö
gru/while/gru_cell_2/BiasAdd_5BiasAdd'gru/while/gru_cell_2/MatMul_5:product:0.gru/while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_5¾
gru/while/gru_cell_2/mul_6Mul"gru/while/gru_cell_2/Sigmoid_1:y:0'gru/while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_6¼
gru/while/gru_cell_2/add_2AddV2'gru/while/gru_cell_2/BiasAdd_2:output:0gru/while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add_2
gru/while/gru_cell_2/TanhTanhgru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/Tanh¬
gru/while/gru_cell_2/mul_7Mul gru/while/gru_cell_2/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_7}
gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell_2/sub/x´
gru/while/gru_cell_2/subSub#gru/while/gru_cell_2/sub/x:output:0 gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/sub®
gru/while/gru_cell_2/mul_8Mulgru/while/gru_cell_2/sub:z:0gru/while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_8³
gru/while/gru_cell_2/add_3AddV2gru/while/gru_cell_2/mul_7:z:0gru/while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add_3
gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile/multiplesµ
gru/while/TileTile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0!gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/Tile¸
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell_2/add_3:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/SelectV2
gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile_1/multiples»
gru/while/Tile_1Tile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0#gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/Tile_1¾
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell_2/add_3:z:0gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/SelectV2_1ï
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/SelectV2:output:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1
gru/while/IdentityIdentitygru/while/add_1:z:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1
gru/while/Identity_2Identitygru/while/add:z:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2¯
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/SelectV2:output:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_4
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_5"b
.gru_while_gru_cell_2_readvariableop_1_resource0gru_while_gru_cell_2_readvariableop_1_resource_0"b
.gru_while_gru_cell_2_readvariableop_4_resource0gru_while_gru_cell_2_readvariableop_4_resource_0"^
,gru_while_gru_cell_2_readvariableop_resource.gru_while_gru_cell_2_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"5
gru_while_identity_5gru/while/Identity_5:output:0"À
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 2J
#gru/while/gru_cell_2/ReadVariableOp#gru/while/gru_cell_2/ReadVariableOp2N
%gru/while/gru_cell_2/ReadVariableOp_1%gru/while/gru_cell_2/ReadVariableOp_12N
%gru/while/gru_cell_2/ReadVariableOp_2%gru/while/gru_cell_2/ReadVariableOp_22N
%gru/while/gru_cell_2/ReadVariableOp_3%gru/while/gru_cell_2/ReadVariableOp_32N
%gru/while/gru_cell_2/ReadVariableOp_4%gru/while/gru_cell_2/ReadVariableOp_42N
%gru/while/gru_cell_2/ReadVariableOp_5%gru/while/gru_cell_2/ReadVariableOp_52N
%gru/while/gru_cell_2/ReadVariableOp_6%gru/while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ
æ
while_body_45359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`?
,while_gru_cell_2_readvariableop_1_resource_0:	¬`>
,while_gru_cell_2_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`=
*while_gru_cell_2_readvariableop_1_resource:	¬`<
*while_gru_cell_2_readvariableop_4_resource: `¢while/gru_cell_2/ReadVariableOp¢!while/gru_cell_2/ReadVariableOp_1¢!while/gru_cell_2/ReadVariableOp_2¢!while/gru_cell_2/ReadVariableOp_3¢!while/gru_cell_2/ReadVariableOp_4¢!while/gru_cell_2/ReadVariableOp_5¢!while/gru_cell_2/ReadVariableOp_6Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¤
 while/gru_cell_2/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell_2/ones_like/ConstÉ
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/ones_like
while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_2/dropout/ConstÄ
while/gru_cell_2/dropout/MulMul#while/gru_cell_2/ones_like:output:0'while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/dropout/Mul
while/gru_cell_2/dropout/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_2/dropout/Shape
5while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Çë27
5while/gru_cell_2/dropout/random_uniform/RandomUniform
'while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell_2/dropout/GreaterEqual/y
%while/gru_cell_2/dropout/GreaterEqualGreaterEqual>while/gru_cell_2/dropout/random_uniform/RandomUniform:output:00while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell_2/dropout/GreaterEqual³
while/gru_cell_2/dropout/CastCast)while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/dropout/Cast¿
while/gru_cell_2/dropout/Mul_1Mul while/gru_cell_2/dropout/Mul:z:0!while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout/Mul_1
 while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_1/ConstÊ
while/gru_cell_2/dropout_1/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout_1/Mul
 while/gru_cell_2/dropout_1/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_1/Shape
7while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Î¯,29
7while/gru_cell_2/dropout_1/random_uniform/RandomUniform
)while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_1/GreaterEqual/y
'while/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'while/gru_cell_2/dropout_1/GreaterEqual¹
while/gru_cell_2/dropout_1/CastCast+while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
while/gru_cell_2/dropout_1/CastÇ
 while/gru_cell_2/dropout_1/Mul_1Mul"while/gru_cell_2/dropout_1/Mul:z:0#while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 while/gru_cell_2/dropout_1/Mul_1
 while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_2/ConstÊ
while/gru_cell_2/dropout_2/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout_2/Mul
 while/gru_cell_2/dropout_2/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_2/Shape
7while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2¸d29
7while/gru_cell_2/dropout_2/random_uniform/RandomUniform
)while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_2/GreaterEqual/y
'while/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'while/gru_cell_2/dropout_2/GreaterEqual¹
while/gru_cell_2/dropout_2/CastCast+while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
while/gru_cell_2/dropout_2/CastÇ
 while/gru_cell_2/dropout_2/Mul_1Mul"while/gru_cell_2/dropout_2/Mul:z:0#while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 while/gru_cell_2/dropout_2/Mul_1
"while/gru_cell_2/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/gru_cell_2/ones_like_1/Shape
"while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/gru_cell_2/ones_like_1/ConstÐ
while/gru_cell_2/ones_like_1Fill+while/gru_cell_2/ones_like_1/Shape:output:0+while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/ones_like_1
 while/gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_3/ConstË
while/gru_cell_2/dropout_3/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_3/Mul
 while/gru_cell_2/dropout_3/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_3/Shape
7while/gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2«29
7while/gru_cell_2/dropout_3/random_uniform/RandomUniform
)while/gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_3/GreaterEqual/y
'while/gru_cell_2/dropout_3/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_3/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_3/GreaterEqual¸
while/gru_cell_2/dropout_3/CastCast+while/gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_3/CastÆ
 while/gru_cell_2/dropout_3/Mul_1Mul"while/gru_cell_2/dropout_3/Mul:z:0#while/gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_3/Mul_1
 while/gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_4/ConstË
while/gru_cell_2/dropout_4/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_4/Mul
 while/gru_cell_2/dropout_4/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_4/Shape
7while/gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ÄÔ29
7while/gru_cell_2/dropout_4/random_uniform/RandomUniform
)while/gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_4/GreaterEqual/y
'while/gru_cell_2/dropout_4/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_4/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_4/GreaterEqual¸
while/gru_cell_2/dropout_4/CastCast+while/gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_4/CastÆ
 while/gru_cell_2/dropout_4/Mul_1Mul"while/gru_cell_2/dropout_4/Mul:z:0#while/gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_4/Mul_1
 while/gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_5/ConstË
while/gru_cell_2/dropout_5/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_5/Mul
 while/gru_cell_2/dropout_5/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_5/Shape
7while/gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2úª³29
7while/gru_cell_2/dropout_5/random_uniform/RandomUniform
)while/gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_5/GreaterEqual/y
'while/gru_cell_2/dropout_5/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_5/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_5/GreaterEqual¸
while/gru_cell_2/dropout_5/CastCast+while/gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_5/CastÆ
 while/gru_cell_2/dropout_5/Mul_1Mul"while/gru_cell_2/dropout_5/Mul:z:0#while/gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_5/Mul_1­
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02!
while/gru_cell_2/ReadVariableOp
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell_2/unstack¼
while/gru_cell_2/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mulÂ
while/gru_cell_2/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_1Â
while/gru_cell_2/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_2´
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_1
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack¡
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice/stack_1¡
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2å
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice±
while/gru_cell_2/MatMulMatMulwhile/gru_cell_2/mul:z:0'while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul´
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_2¡
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_1/stack¥
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_1/stack_1¥
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2ï
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1¹
while/gru_cell_2/MatMul_1MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_1´
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_3¡
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell_2/strided_slice_2/stack¥
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1¥
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2ï
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2¹
while/gru_cell_2/MatMul_2MatMulwhile/gru_cell_2/mul_2:z:0)while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_2
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_3/stack_1
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2Ò
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_3¿
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_4/stack
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_4/stack_1
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2À
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_4Å
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_1
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell_2/strided_slice_5/stack
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2Ð
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2"
 while/gru_cell_2/strided_slice_5Å
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_2¤
while/gru_cell_2/mul_3Mulwhile_placeholder_2$while/gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_3¤
while/gru_cell_2/mul_4Mulwhile_placeholder_2$while/gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_4¤
while/gru_cell_2/mul_5Mulwhile_placeholder_2$while/gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_5³
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_4¡
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack¥
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_6/stack_1¥
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2î
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6¹
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul_3:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_3³
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_5¡
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_7/stack¥
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_7/stack_1¥
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2î
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7¹
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_4:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_4
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_8/stack_1
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2Ò
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_8Å
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_3
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_9/stack
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_9/stack_1
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2À
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_9Å
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_4¯
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoidµ
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_1
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoid_1³
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_6£
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/gru_cell_2/strided_slice_10/stack§
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1§
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2ó
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10º
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_5:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_5
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2)
'while/gru_cell_2/strided_slice_11/stack 
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1 
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2Õ
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2#
!while/gru_cell_2/strided_slice_11Æ
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_5®
while/gru_cell_2/mul_6Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_6¬
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_2
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Tanh
while/gru_cell_2/mul_7Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_7u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_2/sub/x¤
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/sub
while/gru_cell_2/mul_8Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_8£
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_7:z:0while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ø
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityë
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ú
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ô
½
__inference_loss_fn_0_48393S
@gru_gru_cell_2_kernel_regularizer_square_readvariableop_resource:	¬`
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpô
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@gru_gru_cell_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mul¦
IdentityIdentity)gru/gru_cell_2/kernel/Regularizer/mul:z:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp
ÑT
õ
>__inference_gru_layer_call_and_return_conditional_losses_44515

inputs"
gru_cell_2_44427:`#
gru_cell_2_44429:	¬`"
gru_cell_2_44431: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_2/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2ë
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_44427gru_cell_2_44429gru_cell_2_44431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *N
fIRG
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_443722$
"gru_cell_2/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterß
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_44427gru_cell_2_44429gru_cell_2_44431*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44439*
condR
while_cond_44438*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÄ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_2_44429*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mul×
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_2_44431*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mul
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp#^gru_cell_2/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
¼

E__inference_gru_cell_2_layer_call_and_return_conditional_losses_48382

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeÑ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Â2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeÖ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2¾á2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_1/GreaterEqual/yÇ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape×
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Àç±2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_2/GreaterEqual/yÇ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ones_like_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/Const
dropout_3/MulMulones_like_1:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeÖ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2¦©2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_3/GreaterEqual/yÆ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/Mul_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/ShapeÖ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2§¦2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_4/GreaterEqual/yÆ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/ShapeÖ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ý²2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_5/GreaterEqual/yÆ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/Mul_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ÿ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ì
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2Ú
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2ê
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5~
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_3~
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2ì
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2Ú
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stack
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2ï
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3Í
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulà
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÞ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityâ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0

Å
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_46135

inputs8
&gru_gru_cell_2_readvariableop_resource:`;
(gru_gru_cell_2_readvariableop_1_resource:	¬`:
(gru_gru_cell_2_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identity¢gru/gru_cell_2/ReadVariableOp¢gru/gru_cell_2/ReadVariableOp_1¢gru/gru_cell_2/ReadVariableOp_2¢gru/gru_cell_2/ReadVariableOp_3¢gru/gru_cell_2/ReadVariableOp_4¢gru/gru_cell_2/ReadVariableOp_5¢gru/gru_cell_2/ReadVariableOp_6¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢	gru/while¢output/BiasAdd/ReadVariableOp¢output/Tensordot/ReadVariableOpm
masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking/NotEqual/y
masking/NotEqualNotEqualinputsmasking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
masking/NotEqual
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
masking/Any/reduction_indices¦
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
masking/Any
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
masking/mul
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
masking/SqueezeU
	gru/ShapeShapemasking/mul:z:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ú
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm
gru/transpose	Transposemasking/mul:z:0gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1s
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/ExpandDims/dim¤
gru/ExpandDims
ExpandDimsmasking/Squeeze:output:0gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gru/ExpandDims
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm¦
gru/transpose_1	Transposegru/ExpandDims:output:0gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
gru/transpose_1
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
gru/TensorArrayV2/element_shapeÂ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2Ç
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
gru/strided_slice_2
gru/gru_cell_2/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2 
gru/gru_cell_2/ones_like/Shape
gru/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
gru/gru_cell_2/ones_like/ConstÁ
gru/gru_cell_2/ones_likeFill'gru/gru_cell_2/ones_like/Shape:output:0'gru/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/ones_like
 gru/gru_cell_2/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2"
 gru/gru_cell_2/ones_like_1/Shape
 gru/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 gru/gru_cell_2/ones_like_1/ConstÈ
gru/gru_cell_2/ones_like_1Fill)gru/gru_cell_2/ones_like_1/Shape:output:0)gru/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/ones_like_1¥
gru/gru_cell_2/ReadVariableOpReadVariableOp&gru_gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell_2/ReadVariableOp
gru/gru_cell_2/unstackUnpack%gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell_2/unstack£
gru/gru_cell_2/mulMulgru/strided_slice_2:output:0!gru/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/mul§
gru/gru_cell_2/mul_1Mulgru/strided_slice_2:output:0!gru/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/mul_1§
gru/gru_cell_2/mul_2Mulgru/strided_slice_2:output:0!gru/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell_2/mul_2¬
gru/gru_cell_2/ReadVariableOp_1ReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02!
gru/gru_cell_2/ReadVariableOp_1
"gru/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell_2/strided_slice/stack
$gru/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice/stack_1
$gru/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell_2/strided_slice/stack_2Ù
gru/gru_cell_2/strided_sliceStridedSlice'gru/gru_cell_2/ReadVariableOp_1:value:0+gru/gru_cell_2/strided_slice/stack:output:0-gru/gru_cell_2/strided_slice/stack_1:output:0-gru/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell_2/strided_slice©
gru/gru_cell_2/MatMulMatMulgru/gru_cell_2/mul:z:0%gru/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul¬
gru/gru_cell_2/ReadVariableOp_2ReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02!
gru/gru_cell_2/ReadVariableOp_2
$gru/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice_1/stack¡
&gru/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&gru/gru_cell_2/strided_slice_1/stack_1¡
&gru/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_1/stack_2ã
gru/gru_cell_2/strided_slice_1StridedSlice'gru/gru_cell_2/ReadVariableOp_2:value:0-gru/gru_cell_2/strided_slice_1/stack:output:0/gru/gru_cell_2/strided_slice_1/stack_1:output:0/gru/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_1±
gru/gru_cell_2/MatMul_1MatMulgru/gru_cell_2/mul_1:z:0'gru/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_1¬
gru/gru_cell_2/ReadVariableOp_3ReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02!
gru/gru_cell_2/ReadVariableOp_3
$gru/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell_2/strided_slice_2/stack¡
&gru/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/gru_cell_2/strided_slice_2/stack_1¡
&gru/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_2/stack_2ã
gru/gru_cell_2/strided_slice_2StridedSlice'gru/gru_cell_2/ReadVariableOp_3:value:0-gru/gru_cell_2/strided_slice_2/stack:output:0/gru/gru_cell_2/strided_slice_2/stack_1:output:0/gru/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_2±
gru/gru_cell_2/MatMul_2MatMulgru/gru_cell_2/mul_2:z:0'gru/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_2
$gru/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_3/stack
&gru/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&gru/gru_cell_2/strided_slice_3/stack_1
&gru/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_3/stack_2Æ
gru/gru_cell_2/strided_slice_3StridedSlicegru/gru_cell_2/unstack:output:0-gru/gru_cell_2/strided_slice_3/stack:output:0/gru/gru_cell_2/strided_slice_3/stack_1:output:0/gru/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
gru/gru_cell_2/strided_slice_3·
gru/gru_cell_2/BiasAddBiasAddgru/gru_cell_2/MatMul:product:0'gru/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd
$gru/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_4/stack
&gru/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&gru/gru_cell_2/strided_slice_4/stack_1
&gru/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_4/stack_2´
gru/gru_cell_2/strided_slice_4StridedSlicegru/gru_cell_2/unstack:output:0-gru/gru_cell_2/strided_slice_4/stack:output:0/gru/gru_cell_2/strided_slice_4/stack_1:output:0/gru/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
gru/gru_cell_2/strided_slice_4½
gru/gru_cell_2/BiasAdd_1BiasAdd!gru/gru_cell_2/MatMul_1:product:0'gru/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_1
$gru/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell_2/strided_slice_5/stack
&gru/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&gru/gru_cell_2/strided_slice_5/stack_1
&gru/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_5/stack_2Ä
gru/gru_cell_2/strided_slice_5StridedSlicegru/gru_cell_2/unstack:output:0-gru/gru_cell_2/strided_slice_5/stack:output:0/gru/gru_cell_2/strided_slice_5/stack_1:output:0/gru/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
gru/gru_cell_2/strided_slice_5½
gru/gru_cell_2/BiasAdd_2BiasAdd!gru/gru_cell_2/MatMul_2:product:0'gru/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_2
gru/gru_cell_2/mul_3Mulgru/zeros:output:0#gru/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_3
gru/gru_cell_2/mul_4Mulgru/zeros:output:0#gru/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_4
gru/gru_cell_2/mul_5Mulgru/zeros:output:0#gru/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_5«
gru/gru_cell_2/ReadVariableOp_4ReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02!
gru/gru_cell_2/ReadVariableOp_4
$gru/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice_6/stack¡
&gru/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/gru_cell_2/strided_slice_6/stack_1¡
&gru/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_6/stack_2â
gru/gru_cell_2/strided_slice_6StridedSlice'gru/gru_cell_2/ReadVariableOp_4:value:0-gru/gru_cell_2/strided_slice_6/stack:output:0/gru/gru_cell_2/strided_slice_6/stack_1:output:0/gru/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_6±
gru/gru_cell_2/MatMul_3MatMulgru/gru_cell_2/mul_3:z:0'gru/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_3«
gru/gru_cell_2/ReadVariableOp_5ReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02!
gru/gru_cell_2/ReadVariableOp_5
$gru/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell_2/strided_slice_7/stack¡
&gru/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&gru/gru_cell_2/strided_slice_7/stack_1¡
&gru/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/gru_cell_2/strided_slice_7/stack_2â
gru/gru_cell_2/strided_slice_7StridedSlice'gru/gru_cell_2/ReadVariableOp_5:value:0-gru/gru_cell_2/strided_slice_7/stack:output:0/gru/gru_cell_2/strided_slice_7/stack_1:output:0/gru/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
gru/gru_cell_2/strided_slice_7±
gru/gru_cell_2/MatMul_4MatMulgru/gru_cell_2/mul_4:z:0'gru/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_4
$gru/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_8/stack
&gru/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&gru/gru_cell_2/strided_slice_8/stack_1
&gru/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_8/stack_2Æ
gru/gru_cell_2/strided_slice_8StridedSlicegru/gru_cell_2/unstack:output:1-gru/gru_cell_2/strided_slice_8/stack:output:0/gru/gru_cell_2/strided_slice_8/stack_1:output:0/gru/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
gru/gru_cell_2/strided_slice_8½
gru/gru_cell_2/BiasAdd_3BiasAdd!gru/gru_cell_2/MatMul_3:product:0'gru/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_3
$gru/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell_2/strided_slice_9/stack
&gru/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&gru/gru_cell_2/strided_slice_9/stack_1
&gru/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&gru/gru_cell_2/strided_slice_9/stack_2´
gru/gru_cell_2/strided_slice_9StridedSlicegru/gru_cell_2/unstack:output:1-gru/gru_cell_2/strided_slice_9/stack:output:0/gru/gru_cell_2/strided_slice_9/stack_1:output:0/gru/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
gru/gru_cell_2/strided_slice_9½
gru/gru_cell_2/BiasAdd_4BiasAdd!gru/gru_cell_2/MatMul_4:product:0'gru/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_4§
gru/gru_cell_2/addAddV2gru/gru_cell_2/BiasAdd:output:0!gru/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add
gru/gru_cell_2/SigmoidSigmoidgru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/Sigmoid­
gru/gru_cell_2/add_1AddV2!gru/gru_cell_2/BiasAdd_1:output:0!gru/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add_1
gru/gru_cell_2/Sigmoid_1Sigmoidgru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/Sigmoid_1«
gru/gru_cell_2/ReadVariableOp_6ReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02!
gru/gru_cell_2/ReadVariableOp_6
%gru/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%gru/gru_cell_2/strided_slice_10/stack£
'gru/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'gru/gru_cell_2/strided_slice_10/stack_1£
'gru/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'gru/gru_cell_2/strided_slice_10/stack_2ç
gru/gru_cell_2/strided_slice_10StridedSlice'gru/gru_cell_2/ReadVariableOp_6:value:0.gru/gru_cell_2/strided_slice_10/stack:output:00gru/gru_cell_2/strided_slice_10/stack_1:output:00gru/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
gru/gru_cell_2/strided_slice_10²
gru/gru_cell_2/MatMul_5MatMulgru/gru_cell_2/mul_5:z:0(gru/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/MatMul_5
%gru/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%gru/gru_cell_2/strided_slice_11/stack
'gru/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'gru/gru_cell_2/strided_slice_11/stack_1
'gru/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'gru/gru_cell_2/strided_slice_11/stack_2É
gru/gru_cell_2/strided_slice_11StridedSlicegru/gru_cell_2/unstack:output:1.gru/gru_cell_2/strided_slice_11/stack:output:00gru/gru_cell_2/strided_slice_11/stack_1:output:00gru/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
gru/gru_cell_2/strided_slice_11¾
gru/gru_cell_2/BiasAdd_5BiasAdd!gru/gru_cell_2/MatMul_5:product:0(gru/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/BiasAdd_5¦
gru/gru_cell_2/mul_6Mulgru/gru_cell_2/Sigmoid_1:y:0!gru/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_6¤
gru/gru_cell_2/add_2AddV2!gru/gru_cell_2/BiasAdd_2:output:0gru/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add_2~
gru/gru_cell_2/TanhTanhgru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/Tanh
gru/gru_cell_2/mul_7Mulgru/gru_cell_2/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_7q
gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell_2/sub/x
gru/gru_cell_2/subSubgru/gru_cell_2/sub/x:output:0gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/sub
gru/gru_cell_2/mul_8Mulgru/gru_cell_2/sub:z:0gru/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/mul_8
gru/gru_cell_2/add_3AddV2gru/gru_cell_2/mul_7:z:0gru/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell_2/add_3
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2#
!gru/TensorArrayV2_1/element_shapeÈ
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time
!gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/TensorArrayV2_2/element_shapeÈ
gru/TensorArrayV2_2TensorListReserve*gru/TensorArrayV2_2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
gru/TensorArrayV2_2Ë
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shape
-gru/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorgru/transpose_1:y:0Dgru/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02/
-gru/TensorArrayUnstack_1/TensorListFromTensory
gru/zeros_like	ZerosLikegru/gru_cell_2/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/zeros_like
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counterÐ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0&gru_gru_cell_2_readvariableop_resource(gru_gru_cell_2_readvariableop_1_resource(gru_gru_cell_2_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
gru_while_body_45930* 
condR
gru_while_cond_45929*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : *
parallel_iterations 2
	gru/while½
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    26
4gru/TensorArrayV2Stack/TensorListStack/element_shape
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
gru/strided_slice_3/stack
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2²
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gru/strided_slice_3
gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_2/perm¾
gru/transpose_2	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
gru/transpose_2n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime«
output/Tensordot/ReadVariableOpReadVariableOp(output_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02!
output/Tensordot/ReadVariableOpx
output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/axes
output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
output/Tensordot/frees
output/Tensordot/ShapeShapegru/transpose_2:y:0*
T0*
_output_shapes
:2
output/Tensordot/Shape
output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/GatherV2/axisô
output/Tensordot/GatherV2GatherV2output/Tensordot/Shape:output:0output/Tensordot/free:output:0'output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2
 output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 output/Tensordot/GatherV2_1/axisú
output/Tensordot/GatherV2_1GatherV2output/Tensordot/Shape:output:0output/Tensordot/axes:output:0)output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2_1z
output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const
output/Tensordot/ProdProd"output/Tensordot/GatherV2:output:0output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod~
output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
output/Tensordot/Const_1¤
output/Tensordot/Prod_1Prod$output/Tensordot/GatherV2_1:output:0!output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
output/Tensordot/Prod_1~
output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
output/Tensordot/concat/axisÓ
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat¨
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stack»
output/Tensordot/transpose	Transposegru/transpose_2:y:0 output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
output/Tensordot/transpose»
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
output/Tensordot/Reshapeº
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/Tensordot/MatMul~
output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/Const_2
output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/concat_1/axisà
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1µ
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
output/Tensordot¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp¬
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
output/BiasAddÜ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulï
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp(gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mul°
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell_2/ReadVariableOp ^gru/gru_cell_2/ReadVariableOp_1 ^gru/gru_cell_2/ReadVariableOp_2 ^gru/gru_cell_2/ReadVariableOp_3 ^gru/gru_cell_2/ReadVariableOp_4 ^gru/gru_cell_2/ReadVariableOp_5 ^gru/gru_cell_2/ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2>
gru/gru_cell_2/ReadVariableOpgru/gru_cell_2/ReadVariableOp2B
gru/gru_cell_2/ReadVariableOp_1gru/gru_cell_2/ReadVariableOp_12B
gru/gru_cell_2/ReadVariableOp_2gru/gru_cell_2/ReadVariableOp_22B
gru/gru_cell_2/ReadVariableOp_3gru/gru_cell_2/ReadVariableOp_32B
gru/gru_cell_2/ReadVariableOp_4gru/gru_cell_2/ReadVariableOp_42B
gru/gru_cell_2/ReadVariableOp_5gru/gru_cell_2/ReadVariableOp_52B
gru/gru_cell_2/ReadVariableOp_6gru/gru_cell_2/ReadVariableOp_62r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2
	gru/while	gru/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2B
output/Tensordot/ReadVariableOpoutput/Tensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ô
Ü
>__inference_gru_layer_call_and_return_conditional_losses_46950
inputs_04
"gru_cell_2_readvariableop_resource:`7
$gru_cell_2_readvariableop_1_resource:	¬`6
$gru_cell_2_readvariableop_4_resource: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell_2/ReadVariableOp¢gru_cell_2/ReadVariableOp_1¢gru_cell_2/ReadVariableOp_2¢gru_cell_2/ReadVariableOp_3¢gru_cell_2/ReadVariableOp_4¢gru_cell_2/ReadVariableOp_5¢gru_cell_2/ReadVariableOp_6¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2
gru_cell_2/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like/Const±
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/ones_likez
gru_cell_2/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like_1/Shape
gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/ones_like_1/Const¸
gru_cell_2/ones_like_1Fill%gru_cell_2/ones_like_1/Shape:output:0%gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/ones_like_1
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_2/ReadVariableOp
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_2/unstack
gru_cell_2/mulMulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul
gru_cell_2/mul_1Mulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_1
gru_cell_2/mul_2Mulstrided_slice_2:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell_2/mul_2 
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_1
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice/stack_1
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2Á
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice
gru_cell_2/MatMulMatMulgru_cell_2/mul:z:0!gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul 
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_2
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_1/stack
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_1/stack_1
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2Ë
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_1¡
gru_cell_2/MatMul_1MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_1 
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell_2/ReadVariableOp_3
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell_2/strided_slice_2/stack
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2Ë
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell_2/strided_slice_2¡
gru_cell_2/MatMul_2MatMulgru_cell_2/mul_2:z:0#gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_2
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_3/stack_1
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2®
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_3§
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_4/stack
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_4/stack_1
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_4­
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_1
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell_2/strided_slice_5/stack
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2¬
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_5­
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_2
gru_cell_2/mul_3Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_3
gru_cell_2/mul_4Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_4
gru_cell_2/mul_5Mulzeros:output:0gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_5
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_4
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_6/stack_1
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2Ê
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_6¡
gru_cell_2/MatMul_3MatMulgru_cell_2/mul_3:z:0#gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_3
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_5
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_7/stack
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru_cell_2/strided_slice_7/stack_1
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2Ê
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_7¡
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_4:z:0#gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_4
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_8/stack_1
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2®
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell_2/strided_slice_8­
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_3
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_9/stack
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2$
"gru_cell_2/strided_slice_9/stack_1
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell_2/strided_slice_9­
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_4
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/addy
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_1
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Sigmoid_1
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell_2/ReadVariableOp_6
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!gru_cell_2/strided_slice_10/stack
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2Ï
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell_2/strided_slice_10¢
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_5:z:0$gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/MatMul_5
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2#
!gru_cell_2/strided_slice_11/stack
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2±
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell_2/strided_slice_11®
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/BiasAdd_5
gru_cell_2/mul_6Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_6
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_2r
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/Tanh
gru_cell_2/mul_7Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_7i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_2/sub/x
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/sub
gru_cell_2/mul_8Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/mul_8
gru_cell_2/add_3AddV2gru_cell_2/mul_7:z:0gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell_2/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_46786*
condR
while_cond_46785*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulë
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp$gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÆ
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
®
ç
#__inference_signature_wrapper_45753	
input
unknown:`
	unknown_0:	¬`
	unknown_1: `
	unknown_2: 
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8 *)
f$R"
 __inference__wrapped_model_439452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
û
²
#__inference_gru_layer_call_fn_46655

inputs
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_455712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ø
æ
while_body_47129
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`?
,while_gru_cell_2_readvariableop_1_resource_0:	¬`>
,while_gru_cell_2_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`=
*while_gru_cell_2_readvariableop_1_resource:	¬`<
*while_gru_cell_2_readvariableop_4_resource: `¢while/gru_cell_2/ReadVariableOp¢!while/gru_cell_2/ReadVariableOp_1¢!while/gru_cell_2/ReadVariableOp_2¢!while/gru_cell_2/ReadVariableOp_3¢!while/gru_cell_2/ReadVariableOp_4¢!while/gru_cell_2/ReadVariableOp_5¢!while/gru_cell_2/ReadVariableOp_6Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¤
 while/gru_cell_2/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell_2/ones_like/ConstÉ
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/ones_like
while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_2/dropout/ConstÄ
while/gru_cell_2/dropout/MulMul#while/gru_cell_2/ones_like:output:0'while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/dropout/Mul
while/gru_cell_2/dropout/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_2/dropout/Shape
5while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Ùª27
5while/gru_cell_2/dropout/random_uniform/RandomUniform
'while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell_2/dropout/GreaterEqual/y
%while/gru_cell_2/dropout/GreaterEqualGreaterEqual>while/gru_cell_2/dropout/random_uniform/RandomUniform:output:00while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell_2/dropout/GreaterEqual³
while/gru_cell_2/dropout/CastCast)while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/dropout/Cast¿
while/gru_cell_2/dropout/Mul_1Mul while/gru_cell_2/dropout/Mul:z:0!while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout/Mul_1
 while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_1/ConstÊ
while/gru_cell_2/dropout_1/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout_1/Mul
 while/gru_cell_2/dropout_1/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_1/Shape
7while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed229
7while/gru_cell_2/dropout_1/random_uniform/RandomUniform
)while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_1/GreaterEqual/y
'while/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'while/gru_cell_2/dropout_1/GreaterEqual¹
while/gru_cell_2/dropout_1/CastCast+while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
while/gru_cell_2/dropout_1/CastÇ
 while/gru_cell_2/dropout_1/Mul_1Mul"while/gru_cell_2/dropout_1/Mul:z:0#while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 while/gru_cell_2/dropout_1/Mul_1
 while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_2/ConstÊ
while/gru_cell_2/dropout_2/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout_2/Mul
 while/gru_cell_2/dropout_2/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_2/Shape
7while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ø©29
7while/gru_cell_2/dropout_2/random_uniform/RandomUniform
)while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_2/GreaterEqual/y
'while/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'while/gru_cell_2/dropout_2/GreaterEqual¹
while/gru_cell_2/dropout_2/CastCast+while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
while/gru_cell_2/dropout_2/CastÇ
 while/gru_cell_2/dropout_2/Mul_1Mul"while/gru_cell_2/dropout_2/Mul:z:0#while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 while/gru_cell_2/dropout_2/Mul_1
"while/gru_cell_2/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/gru_cell_2/ones_like_1/Shape
"while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/gru_cell_2/ones_like_1/ConstÐ
while/gru_cell_2/ones_like_1Fill+while/gru_cell_2/ones_like_1/Shape:output:0+while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/ones_like_1
 while/gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_3/ConstË
while/gru_cell_2/dropout_3/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_3/Mul
 while/gru_cell_2/dropout_3/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_3/Shape
7while/gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2úáÿ29
7while/gru_cell_2/dropout_3/random_uniform/RandomUniform
)while/gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_3/GreaterEqual/y
'while/gru_cell_2/dropout_3/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_3/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_3/GreaterEqual¸
while/gru_cell_2/dropout_3/CastCast+while/gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_3/CastÆ
 while/gru_cell_2/dropout_3/Mul_1Mul"while/gru_cell_2/dropout_3/Mul:z:0#while/gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_3/Mul_1
 while/gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_4/ConstË
while/gru_cell_2/dropout_4/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_4/Mul
 while/gru_cell_2/dropout_4/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_4/Shape
7while/gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ü®¨29
7while/gru_cell_2/dropout_4/random_uniform/RandomUniform
)while/gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_4/GreaterEqual/y
'while/gru_cell_2/dropout_4/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_4/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_4/GreaterEqual¸
while/gru_cell_2/dropout_4/CastCast+while/gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_4/CastÆ
 while/gru_cell_2/dropout_4/Mul_1Mul"while/gru_cell_2/dropout_4/Mul:z:0#while/gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_4/Mul_1
 while/gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_5/ConstË
while/gru_cell_2/dropout_5/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_5/Mul
 while/gru_cell_2/dropout_5/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_5/Shape
7while/gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2÷È29
7while/gru_cell_2/dropout_5/random_uniform/RandomUniform
)while/gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_5/GreaterEqual/y
'while/gru_cell_2/dropout_5/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_5/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_5/GreaterEqual¸
while/gru_cell_2/dropout_5/CastCast+while/gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_5/CastÆ
 while/gru_cell_2/dropout_5/Mul_1Mul"while/gru_cell_2/dropout_5/Mul:z:0#while/gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_5/Mul_1­
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02!
while/gru_cell_2/ReadVariableOp
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell_2/unstack¼
while/gru_cell_2/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mulÂ
while/gru_cell_2/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_1Â
while/gru_cell_2/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_2´
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_1
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack¡
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice/stack_1¡
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2å
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice±
while/gru_cell_2/MatMulMatMulwhile/gru_cell_2/mul:z:0'while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul´
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_2¡
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_1/stack¥
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_1/stack_1¥
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2ï
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1¹
while/gru_cell_2/MatMul_1MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_1´
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_3¡
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell_2/strided_slice_2/stack¥
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1¥
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2ï
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2¹
while/gru_cell_2/MatMul_2MatMulwhile/gru_cell_2/mul_2:z:0)while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_2
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_3/stack_1
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2Ò
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_3¿
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_4/stack
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_4/stack_1
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2À
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_4Å
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_1
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell_2/strided_slice_5/stack
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2Ð
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2"
 while/gru_cell_2/strided_slice_5Å
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_2¤
while/gru_cell_2/mul_3Mulwhile_placeholder_2$while/gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_3¤
while/gru_cell_2/mul_4Mulwhile_placeholder_2$while/gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_4¤
while/gru_cell_2/mul_5Mulwhile_placeholder_2$while/gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_5³
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_4¡
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack¥
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_6/stack_1¥
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2î
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6¹
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul_3:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_3³
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_5¡
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_7/stack¥
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_7/stack_1¥
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2î
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7¹
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_4:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_4
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_8/stack_1
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2Ò
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_8Å
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_3
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_9/stack
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_9/stack_1
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2À
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_9Å
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_4¯
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoidµ
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_1
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoid_1³
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_6£
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/gru_cell_2/strided_slice_10/stack§
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1§
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2ó
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10º
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_5:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_5
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2)
'while/gru_cell_2/strided_slice_11/stack 
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1 
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2Õ
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2#
!while/gru_cell_2/strided_slice_11Æ
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_5®
while/gru_cell_2/mul_6Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_6¬
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_2
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Tanh
while/gru_cell_2/mul_7Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_7u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_2/sub/x¤
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/sub
while/gru_cell_2/mul_8Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_8£
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_7:z:0while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ø
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityë
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ú
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
çÉ
þ
 __inference__wrapped_model_43945	
inputG
5gru_classifier_gru_gru_cell_2_readvariableop_resource:`J
7gru_classifier_gru_gru_cell_2_readvariableop_1_resource:	¬`I
7gru_classifier_gru_gru_cell_2_readvariableop_4_resource: `I
7gru_classifier_output_tensordot_readvariableop_resource: C
5gru_classifier_output_biasadd_readvariableop_resource:
identity¢,GRU_classifier/gru/gru_cell_2/ReadVariableOp¢.GRU_classifier/gru/gru_cell_2/ReadVariableOp_1¢.GRU_classifier/gru/gru_cell_2/ReadVariableOp_2¢.GRU_classifier/gru/gru_cell_2/ReadVariableOp_3¢.GRU_classifier/gru/gru_cell_2/ReadVariableOp_4¢.GRU_classifier/gru/gru_cell_2/ReadVariableOp_5¢.GRU_classifier/gru/gru_cell_2/ReadVariableOp_6¢GRU_classifier/gru/while¢,GRU_classifier/output/BiasAdd/ReadVariableOp¢.GRU_classifier/output/Tensordot/ReadVariableOp
!GRU_classifier/masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!GRU_classifier/masking/NotEqual/yÁ
GRU_classifier/masking/NotEqualNotEqualinput*GRU_classifier/masking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2!
GRU_classifier/masking/NotEqual§
,GRU_classifier/masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,GRU_classifier/masking/Any/reduction_indicesâ
GRU_classifier/masking/AnyAny#GRU_classifier/masking/NotEqual:z:05GRU_classifier/masking/Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
GRU_classifier/masking/Anyµ
GRU_classifier/masking/CastCast#GRU_classifier/masking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
GRU_classifier/masking/Cast§
GRU_classifier/masking/mulMulinputGRU_classifier/masking/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
GRU_classifier/masking/mulË
GRU_classifier/masking/SqueezeSqueeze#GRU_classifier/masking/Any:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2 
GRU_classifier/masking/Squeeze
GRU_classifier/gru/ShapeShapeGRU_classifier/masking/mul:z:0*
T0*
_output_shapes
:2
GRU_classifier/gru/Shape
&GRU_classifier/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&GRU_classifier/gru/strided_slice/stack
(GRU_classifier/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(GRU_classifier/gru/strided_slice/stack_1
(GRU_classifier/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(GRU_classifier/gru/strided_slice/stack_2Ô
 GRU_classifier/gru/strided_sliceStridedSlice!GRU_classifier/gru/Shape:output:0/GRU_classifier/gru/strided_slice/stack:output:01GRU_classifier/gru/strided_slice/stack_1:output:01GRU_classifier/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 GRU_classifier/gru/strided_slice
GRU_classifier/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2 
GRU_classifier/gru/zeros/mul/y¸
GRU_classifier/gru/zeros/mulMul)GRU_classifier/gru/strided_slice:output:0'GRU_classifier/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_classifier/gru/zeros/mul
GRU_classifier/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2!
GRU_classifier/gru/zeros/Less/y³
GRU_classifier/gru/zeros/LessLess GRU_classifier/gru/zeros/mul:z:0(GRU_classifier/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_classifier/gru/zeros/Less
!GRU_classifier/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!GRU_classifier/gru/zeros/packed/1Ï
GRU_classifier/gru/zeros/packedPack)GRU_classifier/gru/strided_slice:output:0*GRU_classifier/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
GRU_classifier/gru/zeros/packed
GRU_classifier/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
GRU_classifier/gru/zeros/ConstÁ
GRU_classifier/gru/zerosFill(GRU_classifier/gru/zeros/packed:output:0'GRU_classifier/gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
GRU_classifier/gru/zeros
!GRU_classifier/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!GRU_classifier/gru/transpose/permÕ
GRU_classifier/gru/transpose	TransposeGRU_classifier/masking/mul:z:0*GRU_classifier/gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
GRU_classifier/gru/transpose
GRU_classifier/gru/Shape_1Shape GRU_classifier/gru/transpose:y:0*
T0*
_output_shapes
:2
GRU_classifier/gru/Shape_1
(GRU_classifier/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(GRU_classifier/gru/strided_slice_1/stack¢
*GRU_classifier/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_1/stack_1¢
*GRU_classifier/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_1/stack_2à
"GRU_classifier/gru/strided_slice_1StridedSlice#GRU_classifier/gru/Shape_1:output:01GRU_classifier/gru/strided_slice_1/stack:output:03GRU_classifier/gru/strided_slice_1/stack_1:output:03GRU_classifier/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_1
!GRU_classifier/gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!GRU_classifier/gru/ExpandDims/dimà
GRU_classifier/gru/ExpandDims
ExpandDims'GRU_classifier/masking/Squeeze:output:0*GRU_classifier/gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
GRU_classifier/gru/ExpandDims
#GRU_classifier/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#GRU_classifier/gru/transpose_1/permâ
GRU_classifier/gru/transpose_1	Transpose&GRU_classifier/gru/ExpandDims:output:0,GRU_classifier/gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
GRU_classifier/gru/transpose_1«
.GRU_classifier/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.GRU_classifier/gru/TensorArrayV2/element_shapeþ
 GRU_classifier/gru/TensorArrayV2TensorListReserve7GRU_classifier/gru/TensorArrayV2/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 GRU_classifier/gru/TensorArrayV2å
HGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  2J
HGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeÄ
:GRU_classifier/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor GRU_classifier/gru/transpose:y:0QGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:GRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor
(GRU_classifier/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(GRU_classifier/gru/strided_slice_2/stack¢
*GRU_classifier/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_2/stack_1¢
*GRU_classifier/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_2/stack_2ï
"GRU_classifier/gru/strided_slice_2StridedSlice GRU_classifier/gru/transpose:y:01GRU_classifier/gru/strided_slice_2/stack:output:03GRU_classifier/gru/strided_slice_2/stack_1:output:03GRU_classifier/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_2¹
-GRU_classifier/gru/gru_cell_2/ones_like/ShapeShape+GRU_classifier/gru/strided_slice_2:output:0*
T0*
_output_shapes
:2/
-GRU_classifier/gru/gru_cell_2/ones_like/Shape£
-GRU_classifier/gru/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-GRU_classifier/gru/gru_cell_2/ones_like/Constý
'GRU_classifier/gru/gru_cell_2/ones_likeFill6GRU_classifier/gru/gru_cell_2/ones_like/Shape:output:06GRU_classifier/gru/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'GRU_classifier/gru/gru_cell_2/ones_like³
/GRU_classifier/gru/gru_cell_2/ones_like_1/ShapeShape!GRU_classifier/gru/zeros:output:0*
T0*
_output_shapes
:21
/GRU_classifier/gru/gru_cell_2/ones_like_1/Shape§
/GRU_classifier/gru/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/GRU_classifier/gru/gru_cell_2/ones_like_1/Const
)GRU_classifier/gru/gru_cell_2/ones_like_1Fill8GRU_classifier/gru/gru_cell_2/ones_like_1/Shape:output:08GRU_classifier/gru/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/gru_cell_2/ones_like_1Ò
,GRU_classifier/gru/gru_cell_2/ReadVariableOpReadVariableOp5gru_classifier_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype02.
,GRU_classifier/gru/gru_cell_2/ReadVariableOpÄ
%GRU_classifier/gru/gru_cell_2/unstackUnpack4GRU_classifier/gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2'
%GRU_classifier/gru/gru_cell_2/unstackß
!GRU_classifier/gru/gru_cell_2/mulMul+GRU_classifier/gru/strided_slice_2:output:00GRU_classifier/gru/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!GRU_classifier/gru/gru_cell_2/mulã
#GRU_classifier/gru/gru_cell_2/mul_1Mul+GRU_classifier/gru/strided_slice_2:output:00GRU_classifier/gru/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#GRU_classifier/gru/gru_cell_2/mul_1ã
#GRU_classifier/gru/gru_cell_2/mul_2Mul+GRU_classifier/gru/strided_slice_2:output:00GRU_classifier/gru/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#GRU_classifier/gru/gru_cell_2/mul_2Ù
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_1ReadVariableOp7gru_classifier_gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype020
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_1·
1GRU_classifier/gru/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell_2/strided_slice/stack»
3GRU_classifier/gru/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell_2/strided_slice/stack_1»
3GRU_classifier/gru/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell_2/strided_slice/stack_2³
+GRU_classifier/gru/gru_cell_2/strided_sliceStridedSlice6GRU_classifier/gru/gru_cell_2/ReadVariableOp_1:value:0:GRU_classifier/gru/gru_cell_2/strided_slice/stack:output:0<GRU_classifier/gru/gru_cell_2/strided_slice/stack_1:output:0<GRU_classifier/gru/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell_2/strided_sliceå
$GRU_classifier/gru/gru_cell_2/MatMulMatMul%GRU_classifier/gru/gru_cell_2/mul:z:04GRU_classifier/gru/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$GRU_classifier/gru/gru_cell_2/MatMulÙ
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_2ReadVariableOp7gru_classifier_gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype020
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_2»
3GRU_classifier/gru/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell_2/strided_slice_1/stack¿
5GRU_classifier/gru/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   27
5GRU_classifier/gru/gru_cell_2/strided_slice_1/stack_1¿
5GRU_classifier/gru/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5GRU_classifier/gru/gru_cell_2/strided_slice_1/stack_2½
-GRU_classifier/gru/gru_cell_2/strided_slice_1StridedSlice6GRU_classifier/gru/gru_cell_2/ReadVariableOp_2:value:0<GRU_classifier/gru/gru_cell_2/strided_slice_1/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_1/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_1í
&GRU_classifier/gru/gru_cell_2/MatMul_1MatMul'GRU_classifier/gru/gru_cell_2/mul_1:z:06GRU_classifier/gru/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&GRU_classifier/gru/gru_cell_2/MatMul_1Ù
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_3ReadVariableOp7gru_classifier_gru_gru_cell_2_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype020
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_3»
3GRU_classifier/gru/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell_2/strided_slice_2/stack¿
5GRU_classifier/gru/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5GRU_classifier/gru/gru_cell_2/strided_slice_2/stack_1¿
5GRU_classifier/gru/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5GRU_classifier/gru/gru_cell_2/strided_slice_2/stack_2½
-GRU_classifier/gru/gru_cell_2/strided_slice_2StridedSlice6GRU_classifier/gru/gru_cell_2/ReadVariableOp_3:value:0<GRU_classifier/gru/gru_cell_2/strided_slice_2/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_2/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_2í
&GRU_classifier/gru/gru_cell_2/MatMul_2MatMul'GRU_classifier/gru/gru_cell_2/mul_2:z:06GRU_classifier/gru/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&GRU_classifier/gru/gru_cell_2/MatMul_2´
3GRU_classifier/gru/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell_2/strided_slice_3/stack¸
5GRU_classifier/gru/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5GRU_classifier/gru/gru_cell_2/strided_slice_3/stack_1¸
5GRU_classifier/gru/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5GRU_classifier/gru/gru_cell_2/strided_slice_3/stack_2 
-GRU_classifier/gru/gru_cell_2/strided_slice_3StridedSlice.GRU_classifier/gru/gru_cell_2/unstack:output:0<GRU_classifier/gru/gru_cell_2/strided_slice_3/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_3/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_3ó
%GRU_classifier/gru/gru_cell_2/BiasAddBiasAdd.GRU_classifier/gru/gru_cell_2/MatMul:product:06GRU_classifier/gru/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell_2/BiasAdd´
3GRU_classifier/gru/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell_2/strided_slice_4/stack¸
5GRU_classifier/gru/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@27
5GRU_classifier/gru/gru_cell_2/strided_slice_4/stack_1¸
5GRU_classifier/gru/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5GRU_classifier/gru/gru_cell_2/strided_slice_4/stack_2
-GRU_classifier/gru/gru_cell_2/strided_slice_4StridedSlice.GRU_classifier/gru/gru_cell_2/unstack:output:0<GRU_classifier/gru/gru_cell_2/strided_slice_4/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_4/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2/
-GRU_classifier/gru/gru_cell_2/strided_slice_4ù
'GRU_classifier/gru/gru_cell_2/BiasAdd_1BiasAdd0GRU_classifier/gru/gru_cell_2/MatMul_1:product:06GRU_classifier/gru/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell_2/BiasAdd_1´
3GRU_classifier/gru/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell_2/strided_slice_5/stack¸
5GRU_classifier/gru/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5GRU_classifier/gru/gru_cell_2/strided_slice_5/stack_1¸
5GRU_classifier/gru/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5GRU_classifier/gru/gru_cell_2/strided_slice_5/stack_2
-GRU_classifier/gru/gru_cell_2/strided_slice_5StridedSlice.GRU_classifier/gru/gru_cell_2/unstack:output:0<GRU_classifier/gru/gru_cell_2/strided_slice_5/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_5/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_5ù
'GRU_classifier/gru/gru_cell_2/BiasAdd_2BiasAdd0GRU_classifier/gru/gru_cell_2/MatMul_2:product:06GRU_classifier/gru/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell_2/BiasAdd_2Ú
#GRU_classifier/gru/gru_cell_2/mul_3Mul!GRU_classifier/gru/zeros:output:02GRU_classifier/gru/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/mul_3Ú
#GRU_classifier/gru/gru_cell_2/mul_4Mul!GRU_classifier/gru/zeros:output:02GRU_classifier/gru/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/mul_4Ú
#GRU_classifier/gru/gru_cell_2/mul_5Mul!GRU_classifier/gru/zeros:output:02GRU_classifier/gru/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/mul_5Ø
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_4ReadVariableOp7gru_classifier_gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype020
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_4»
3GRU_classifier/gru/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell_2/strided_slice_6/stack¿
5GRU_classifier/gru/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5GRU_classifier/gru/gru_cell_2/strided_slice_6/stack_1¿
5GRU_classifier/gru/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5GRU_classifier/gru/gru_cell_2/strided_slice_6/stack_2¼
-GRU_classifier/gru/gru_cell_2/strided_slice_6StridedSlice6GRU_classifier/gru/gru_cell_2/ReadVariableOp_4:value:0<GRU_classifier/gru/gru_cell_2/strided_slice_6/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_6/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_6í
&GRU_classifier/gru/gru_cell_2/MatMul_3MatMul'GRU_classifier/gru/gru_cell_2/mul_3:z:06GRU_classifier/gru/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&GRU_classifier/gru/gru_cell_2/MatMul_3Ø
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_5ReadVariableOp7gru_classifier_gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype020
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_5»
3GRU_classifier/gru/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell_2/strided_slice_7/stack¿
5GRU_classifier/gru/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   27
5GRU_classifier/gru/gru_cell_2/strided_slice_7/stack_1¿
5GRU_classifier/gru/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5GRU_classifier/gru/gru_cell_2/strided_slice_7/stack_2¼
-GRU_classifier/gru/gru_cell_2/strided_slice_7StridedSlice6GRU_classifier/gru/gru_cell_2/ReadVariableOp_5:value:0<GRU_classifier/gru/gru_cell_2/strided_slice_7/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_7/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_7í
&GRU_classifier/gru/gru_cell_2/MatMul_4MatMul'GRU_classifier/gru/gru_cell_2/mul_4:z:06GRU_classifier/gru/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&GRU_classifier/gru/gru_cell_2/MatMul_4´
3GRU_classifier/gru/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell_2/strided_slice_8/stack¸
5GRU_classifier/gru/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5GRU_classifier/gru/gru_cell_2/strided_slice_8/stack_1¸
5GRU_classifier/gru/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5GRU_classifier/gru/gru_cell_2/strided_slice_8/stack_2 
-GRU_classifier/gru/gru_cell_2/strided_slice_8StridedSlice.GRU_classifier/gru/gru_cell_2/unstack:output:1<GRU_classifier/gru/gru_cell_2/strided_slice_8/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_8/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2/
-GRU_classifier/gru/gru_cell_2/strided_slice_8ù
'GRU_classifier/gru/gru_cell_2/BiasAdd_3BiasAdd0GRU_classifier/gru/gru_cell_2/MatMul_3:product:06GRU_classifier/gru/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell_2/BiasAdd_3´
3GRU_classifier/gru/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell_2/strided_slice_9/stack¸
5GRU_classifier/gru/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@27
5GRU_classifier/gru/gru_cell_2/strided_slice_9/stack_1¸
5GRU_classifier/gru/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5GRU_classifier/gru/gru_cell_2/strided_slice_9/stack_2
-GRU_classifier/gru/gru_cell_2/strided_slice_9StridedSlice.GRU_classifier/gru/gru_cell_2/unstack:output:1<GRU_classifier/gru/gru_cell_2/strided_slice_9/stack:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_9/stack_1:output:0>GRU_classifier/gru/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2/
-GRU_classifier/gru/gru_cell_2/strided_slice_9ù
'GRU_classifier/gru/gru_cell_2/BiasAdd_4BiasAdd0GRU_classifier/gru/gru_cell_2/MatMul_4:product:06GRU_classifier/gru/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell_2/BiasAdd_4ã
!GRU_classifier/gru/gru_cell_2/addAddV2.GRU_classifier/gru/gru_cell_2/BiasAdd:output:00GRU_classifier/gru/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell_2/add²
%GRU_classifier/gru/gru_cell_2/SigmoidSigmoid%GRU_classifier/gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell_2/Sigmoidé
#GRU_classifier/gru/gru_cell_2/add_1AddV20GRU_classifier/gru/gru_cell_2/BiasAdd_1:output:00GRU_classifier/gru/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/add_1¸
'GRU_classifier/gru/gru_cell_2/Sigmoid_1Sigmoid'GRU_classifier/gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell_2/Sigmoid_1Ø
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_6ReadVariableOp7gru_classifier_gru_gru_cell_2_readvariableop_4_resource*
_output_shapes

: `*
dtype020
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_6½
4GRU_classifier/gru/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   26
4GRU_classifier/gru/gru_cell_2/strided_slice_10/stackÁ
6GRU_classifier/gru/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6GRU_classifier/gru/gru_cell_2/strided_slice_10/stack_1Á
6GRU_classifier/gru/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6GRU_classifier/gru/gru_cell_2/strided_slice_10/stack_2Á
.GRU_classifier/gru/gru_cell_2/strided_slice_10StridedSlice6GRU_classifier/gru/gru_cell_2/ReadVariableOp_6:value:0=GRU_classifier/gru/gru_cell_2/strided_slice_10/stack:output:0?GRU_classifier/gru/gru_cell_2/strided_slice_10/stack_1:output:0?GRU_classifier/gru/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask20
.GRU_classifier/gru/gru_cell_2/strided_slice_10î
&GRU_classifier/gru/gru_cell_2/MatMul_5MatMul'GRU_classifier/gru/gru_cell_2/mul_5:z:07GRU_classifier/gru/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&GRU_classifier/gru/gru_cell_2/MatMul_5¶
4GRU_classifier/gru/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@26
4GRU_classifier/gru/gru_cell_2/strided_slice_11/stackº
6GRU_classifier/gru/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6GRU_classifier/gru/gru_cell_2/strided_slice_11/stack_1º
6GRU_classifier/gru/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6GRU_classifier/gru/gru_cell_2/strided_slice_11/stack_2£
.GRU_classifier/gru/gru_cell_2/strided_slice_11StridedSlice.GRU_classifier/gru/gru_cell_2/unstack:output:1=GRU_classifier/gru/gru_cell_2/strided_slice_11/stack:output:0?GRU_classifier/gru/gru_cell_2/strided_slice_11/stack_1:output:0?GRU_classifier/gru/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask20
.GRU_classifier/gru/gru_cell_2/strided_slice_11ú
'GRU_classifier/gru/gru_cell_2/BiasAdd_5BiasAdd0GRU_classifier/gru/gru_cell_2/MatMul_5:product:07GRU_classifier/gru/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell_2/BiasAdd_5â
#GRU_classifier/gru/gru_cell_2/mul_6Mul+GRU_classifier/gru/gru_cell_2/Sigmoid_1:y:00GRU_classifier/gru/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/mul_6à
#GRU_classifier/gru/gru_cell_2/add_2AddV20GRU_classifier/gru/gru_cell_2/BiasAdd_2:output:0'GRU_classifier/gru/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/add_2«
"GRU_classifier/gru/gru_cell_2/TanhTanh'GRU_classifier/gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"GRU_classifier/gru/gru_cell_2/TanhÑ
#GRU_classifier/gru/gru_cell_2/mul_7Mul)GRU_classifier/gru/gru_cell_2/Sigmoid:y:0!GRU_classifier/gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/mul_7
#GRU_classifier/gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#GRU_classifier/gru/gru_cell_2/sub/xØ
!GRU_classifier/gru/gru_cell_2/subSub,GRU_classifier/gru/gru_cell_2/sub/x:output:0)GRU_classifier/gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell_2/subÒ
#GRU_classifier/gru/gru_cell_2/mul_8Mul%GRU_classifier/gru/gru_cell_2/sub:z:0&GRU_classifier/gru/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/mul_8×
#GRU_classifier/gru/gru_cell_2/add_3AddV2'GRU_classifier/gru/gru_cell_2/mul_7:z:0'GRU_classifier/gru/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell_2/add_3µ
0GRU_classifier/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0GRU_classifier/gru/TensorArrayV2_1/element_shape
"GRU_classifier/gru/TensorArrayV2_1TensorListReserve9GRU_classifier/gru/TensorArrayV2_1/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"GRU_classifier/gru/TensorArrayV2_1t
GRU_classifier/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_classifier/gru/time¯
0GRU_classifier/gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0GRU_classifier/gru/TensorArrayV2_2/element_shape
"GRU_classifier/gru/TensorArrayV2_2TensorListReserve9GRU_classifier/gru/TensorArrayV2_2/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02$
"GRU_classifier/gru/TensorArrayV2_2é
JGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2L
JGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeÌ
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor"GRU_classifier/gru/transpose_1:y:0SGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02>
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor¦
GRU_classifier/gru/zeros_like	ZerosLike'GRU_classifier/gru/gru_cell_2/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
GRU_classifier/gru/zeros_like¥
+GRU_classifier/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+GRU_classifier/gru/while/maximum_iterations
%GRU_classifier/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%GRU_classifier/gru/while/loop_counterÀ
GRU_classifier/gru/whileWhile.GRU_classifier/gru/while/loop_counter:output:04GRU_classifier/gru/while/maximum_iterations:output:0 GRU_classifier/gru/time:output:0+GRU_classifier/gru/TensorArrayV2_1:handle:0!GRU_classifier/gru/zeros_like:y:0!GRU_classifier/gru/zeros:output:0+GRU_classifier/gru/strided_slice_1:output:0JGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0LGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:05gru_classifier_gru_gru_cell_2_readvariableop_resource7gru_classifier_gru_gru_cell_2_readvariableop_1_resource7gru_classifier_gru_gru_cell_2_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#GRU_classifier_gru_while_body_43752*/
cond'R%
#GRU_classifier_gru_while_cond_43751*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : *
parallel_iterations 2
GRU_classifier/gru/whileÛ
CGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2E
CGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shape½
5GRU_classifier/gru/TensorArrayV2Stack/TensorListStackTensorListStack!GRU_classifier/gru/while:output:3LGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype027
5GRU_classifier/gru/TensorArrayV2Stack/TensorListStack§
(GRU_classifier/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2*
(GRU_classifier/gru/strided_slice_3/stack¢
*GRU_classifier/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*GRU_classifier/gru/strided_slice_3/stack_1¢
*GRU_classifier/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_3/stack_2
"GRU_classifier/gru/strided_slice_3StridedSlice>GRU_classifier/gru/TensorArrayV2Stack/TensorListStack:tensor:01GRU_classifier/gru/strided_slice_3/stack:output:03GRU_classifier/gru/strided_slice_3/stack_1:output:03GRU_classifier/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_3
#GRU_classifier/gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#GRU_classifier/gru/transpose_2/permú
GRU_classifier/gru/transpose_2	Transpose>GRU_classifier/gru/TensorArrayV2Stack/TensorListStack:tensor:0,GRU_classifier/gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2 
GRU_classifier/gru/transpose_2
GRU_classifier/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_classifier/gru/runtimeØ
.GRU_classifier/output/Tensordot/ReadVariableOpReadVariableOp7gru_classifier_output_tensordot_readvariableop_resource*
_output_shapes

: *
dtype020
.GRU_classifier/output/Tensordot/ReadVariableOp
$GRU_classifier/output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$GRU_classifier/output/Tensordot/axes
$GRU_classifier/output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$GRU_classifier/output/Tensordot/free 
%GRU_classifier/output/Tensordot/ShapeShape"GRU_classifier/gru/transpose_2:y:0*
T0*
_output_shapes
:2'
%GRU_classifier/output/Tensordot/Shape 
-GRU_classifier/output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-GRU_classifier/output/Tensordot/GatherV2/axis¿
(GRU_classifier/output/Tensordot/GatherV2GatherV2.GRU_classifier/output/Tensordot/Shape:output:0-GRU_classifier/output/Tensordot/free:output:06GRU_classifier/output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(GRU_classifier/output/Tensordot/GatherV2¤
/GRU_classifier/output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/GRU_classifier/output/Tensordot/GatherV2_1/axisÅ
*GRU_classifier/output/Tensordot/GatherV2_1GatherV2.GRU_classifier/output/Tensordot/Shape:output:0-GRU_classifier/output/Tensordot/axes:output:08GRU_classifier/output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*GRU_classifier/output/Tensordot/GatherV2_1
%GRU_classifier/output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%GRU_classifier/output/Tensordot/ConstØ
$GRU_classifier/output/Tensordot/ProdProd1GRU_classifier/output/Tensordot/GatherV2:output:0.GRU_classifier/output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$GRU_classifier/output/Tensordot/Prod
'GRU_classifier/output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'GRU_classifier/output/Tensordot/Const_1à
&GRU_classifier/output/Tensordot/Prod_1Prod3GRU_classifier/output/Tensordot/GatherV2_1:output:00GRU_classifier/output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&GRU_classifier/output/Tensordot/Prod_1
+GRU_classifier/output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+GRU_classifier/output/Tensordot/concat/axis
&GRU_classifier/output/Tensordot/concatConcatV2-GRU_classifier/output/Tensordot/free:output:0-GRU_classifier/output/Tensordot/axes:output:04GRU_classifier/output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&GRU_classifier/output/Tensordot/concatä
%GRU_classifier/output/Tensordot/stackPack-GRU_classifier/output/Tensordot/Prod:output:0/GRU_classifier/output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%GRU_classifier/output/Tensordot/stack÷
)GRU_classifier/output/Tensordot/transpose	Transpose"GRU_classifier/gru/transpose_2:y:0/GRU_classifier/output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/output/Tensordot/transpose÷
'GRU_classifier/output/Tensordot/ReshapeReshape-GRU_classifier/output/Tensordot/transpose:y:0.GRU_classifier/output/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'GRU_classifier/output/Tensordot/Reshapeö
&GRU_classifier/output/Tensordot/MatMulMatMul0GRU_classifier/output/Tensordot/Reshape:output:06GRU_classifier/output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&GRU_classifier/output/Tensordot/MatMul
'GRU_classifier/output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'GRU_classifier/output/Tensordot/Const_2 
-GRU_classifier/output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-GRU_classifier/output/Tensordot/concat_1/axis«
(GRU_classifier/output/Tensordot/concat_1ConcatV21GRU_classifier/output/Tensordot/GatherV2:output:00GRU_classifier/output/Tensordot/Const_2:output:06GRU_classifier/output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(GRU_classifier/output/Tensordot/concat_1ñ
GRU_classifier/output/TensordotReshape0GRU_classifier/output/Tensordot/MatMul:product:01GRU_classifier/output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2!
GRU_classifier/output/TensordotÎ
,GRU_classifier/output/BiasAdd/ReadVariableOpReadVariableOp5gru_classifier_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,GRU_classifier/output/BiasAdd/ReadVariableOpè
GRU_classifier/output/BiasAddBiasAdd(GRU_classifier/output/Tensordot:output:04GRU_classifier/output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
GRU_classifier/output/BiasAdd×
IdentityIdentity&GRU_classifier/output/BiasAdd:output:0-^GRU_classifier/gru/gru_cell_2/ReadVariableOp/^GRU_classifier/gru/gru_cell_2/ReadVariableOp_1/^GRU_classifier/gru/gru_cell_2/ReadVariableOp_2/^GRU_classifier/gru/gru_cell_2/ReadVariableOp_3/^GRU_classifier/gru/gru_cell_2/ReadVariableOp_4/^GRU_classifier/gru/gru_cell_2/ReadVariableOp_5/^GRU_classifier/gru/gru_cell_2/ReadVariableOp_6^GRU_classifier/gru/while-^GRU_classifier/output/BiasAdd/ReadVariableOp/^GRU_classifier/output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2\
,GRU_classifier/gru/gru_cell_2/ReadVariableOp,GRU_classifier/gru/gru_cell_2/ReadVariableOp2`
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_1.GRU_classifier/gru/gru_cell_2/ReadVariableOp_12`
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_2.GRU_classifier/gru/gru_cell_2/ReadVariableOp_22`
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_3.GRU_classifier/gru/gru_cell_2/ReadVariableOp_32`
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_4.GRU_classifier/gru/gru_cell_2/ReadVariableOp_42`
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_5.GRU_classifier/gru/gru_cell_2/ReadVariableOp_52`
.GRU_classifier/gru/gru_cell_2/ReadVariableOp_6.GRU_classifier/gru/gru_cell_2/ReadVariableOp_624
GRU_classifier/gru/whileGRU_classifier/gru/while2\
,GRU_classifier/output/BiasAdd/ReadVariableOp,GRU_classifier/output/BiasAdd/ReadVariableOp2`
.GRU_classifier/output/Tensordot/ReadVariableOp.GRU_classifier/output/Tensordot/ReadVariableOp:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
ª

Ë
gru_while_cond_46329$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_46329___redundant_placeholder0;
7gru_while_gru_while_cond_46329___redundant_placeholder1;
7gru_while_gru_while_cond_46329___redundant_placeholder2;
7gru_while_gru_while_cond_46329___redundant_placeholder3;
7gru_while_gru_while_cond_46329___redundant_placeholder4
gru_while_identity

gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
À
Æ
#GRU_classifier_gru_while_body_43752B
>gru_classifier_gru_while_gru_classifier_gru_while_loop_counterH
Dgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations(
$gru_classifier_gru_while_placeholder*
&gru_classifier_gru_while_placeholder_1*
&gru_classifier_gru_while_placeholder_2*
&gru_classifier_gru_while_placeholder_3A
=gru_classifier_gru_while_gru_classifier_gru_strided_slice_1_0}
ygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0
}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0O
=gru_classifier_gru_while_gru_cell_2_readvariableop_resource_0:`R
?gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource_0:	¬`Q
?gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource_0: `%
!gru_classifier_gru_while_identity'
#gru_classifier_gru_while_identity_1'
#gru_classifier_gru_while_identity_2'
#gru_classifier_gru_while_identity_3'
#gru_classifier_gru_while_identity_4'
#gru_classifier_gru_while_identity_5?
;gru_classifier_gru_while_gru_classifier_gru_strided_slice_1{
wgru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor
{gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensorM
;gru_classifier_gru_while_gru_cell_2_readvariableop_resource:`P
=gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource:	¬`O
=gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource: `¢2GRU_classifier/gru/while/gru_cell_2/ReadVariableOp¢4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_1¢4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_2¢4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_3¢4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_4¢4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_5¢4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6é
JGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  2L
JGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeÆ
<GRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0$gru_classifier_gru_while_placeholderSGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02>
<GRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItemí
LGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2N
LGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeÏ
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0$gru_classifier_gru_while_placeholderUGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2@
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemÝ
3GRU_classifier/gru/while/gru_cell_2/ones_like/ShapeShapeCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:25
3GRU_classifier/gru/while/gru_cell_2/ones_like/Shape¯
3GRU_classifier/gru/while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3GRU_classifier/gru/while/gru_cell_2/ones_like/Const
-GRU_classifier/gru/while/gru_cell_2/ones_likeFill<GRU_classifier/gru/while/gru_cell_2/ones_like/Shape:output:0<GRU_classifier/gru/while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2/
-GRU_classifier/gru/while/gru_cell_2/ones_likeÄ
5GRU_classifier/gru/while/gru_cell_2/ones_like_1/ShapeShape&gru_classifier_gru_while_placeholder_3*
T0*
_output_shapes
:27
5GRU_classifier/gru/while/gru_cell_2/ones_like_1/Shape³
5GRU_classifier/gru/while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5GRU_classifier/gru/while/gru_cell_2/ones_like_1/Const
/GRU_classifier/gru/while/gru_cell_2/ones_like_1Fill>GRU_classifier/gru/while/gru_cell_2/ones_like_1/Shape:output:0>GRU_classifier/gru/while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/GRU_classifier/gru/while/gru_cell_2/ones_like_1æ
2GRU_classifier/gru/while/gru_cell_2/ReadVariableOpReadVariableOp=gru_classifier_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype024
2GRU_classifier/gru/while/gru_cell_2/ReadVariableOpÖ
+GRU_classifier/gru/while/gru_cell_2/unstackUnpack:GRU_classifier/gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2-
+GRU_classifier/gru/while/gru_cell_2/unstack
'GRU_classifier/gru/while/gru_cell_2/mulMulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:06GRU_classifier/gru/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'GRU_classifier/gru/while/gru_cell_2/mul
)GRU_classifier/gru/while/gru_cell_2/mul_1MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:06GRU_classifier/gru/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2+
)GRU_classifier/gru/while/gru_cell_2/mul_1
)GRU_classifier/gru/while/gru_cell_2/mul_2MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:06GRU_classifier/gru/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2+
)GRU_classifier/gru/while/gru_cell_2/mul_2í
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_1ReadVariableOp?gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype026
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_1Ã
7GRU_classifier/gru/while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell_2/strided_slice/stackÇ
9GRU_classifier/gru/while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice/stack_1Ç
9GRU_classifier/gru/while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice/stack_2×
1GRU_classifier/gru/while/gru_cell_2/strided_sliceStridedSlice<GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_1:value:0@GRU_classifier/gru/while/gru_cell_2/strided_slice/stack:output:0BGRU_classifier/gru/while/gru_cell_2/strided_slice/stack_1:output:0BGRU_classifier/gru/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell_2/strided_sliceý
*GRU_classifier/gru/while/gru_cell_2/MatMulMatMul+GRU_classifier/gru/while/gru_cell_2/mul:z:0:GRU_classifier/gru/while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*GRU_classifier/gru/while/gru_cell_2/MatMulí
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_2ReadVariableOp?gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype026
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_2Ç
9GRU_classifier/gru/while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_1/stackË
;GRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack_1Ë
;GRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack_2á
3GRU_classifier/gru/while/gru_cell_2/strided_slice_1StridedSlice<GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_2:value:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_1
,GRU_classifier/gru/while/gru_cell_2/MatMul_1MatMul-GRU_classifier/gru/while/gru_cell_2/mul_1:z:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,GRU_classifier/gru/while/gru_cell_2/MatMul_1í
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_3ReadVariableOp?gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype026
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_3Ç
9GRU_classifier/gru/while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_2/stackË
;GRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack_1Ë
;GRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack_2á
3GRU_classifier/gru/while/gru_cell_2/strided_slice_2StridedSlice<GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_3:value:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_2
,GRU_classifier/gru/while/gru_cell_2/MatMul_2MatMul-GRU_classifier/gru/while/gru_cell_2/mul_2:z:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,GRU_classifier/gru/while/gru_cell_2/MatMul_2À
9GRU_classifier/gru/while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_3/stackÄ
;GRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack_1Ä
;GRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack_2Ä
3GRU_classifier/gru/while/gru_cell_2/strided_slice_3StridedSlice4GRU_classifier/gru/while/gru_cell_2/unstack:output:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_3
+GRU_classifier/gru/while/gru_cell_2/BiasAddBiasAdd4GRU_classifier/gru/while/gru_cell_2/MatMul:product:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell_2/BiasAddÀ
9GRU_classifier/gru/while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_4/stackÄ
;GRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack_1Ä
;GRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack_2²
3GRU_classifier/gru/while/gru_cell_2/strided_slice_4StridedSlice4GRU_classifier/gru/while/gru_cell_2/unstack:output:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_4
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_1BiasAdd6GRU_classifier/gru/while/gru_cell_2/MatMul_1:product:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_1À
9GRU_classifier/gru/while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_5/stackÄ
;GRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack_1Ä
;GRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack_2Â
3GRU_classifier/gru/while/gru_cell_2/strided_slice_5StridedSlice4GRU_classifier/gru/while/gru_cell_2/unstack:output:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_5
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_2BiasAdd6GRU_classifier/gru/while/gru_cell_2/MatMul_2:product:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_2ñ
)GRU_classifier/gru/while/gru_cell_2/mul_3Mul&gru_classifier_gru_while_placeholder_38GRU_classifier/gru/while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/mul_3ñ
)GRU_classifier/gru/while/gru_cell_2/mul_4Mul&gru_classifier_gru_while_placeholder_38GRU_classifier/gru/while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/mul_4ñ
)GRU_classifier/gru/while/gru_cell_2/mul_5Mul&gru_classifier_gru_while_placeholder_38GRU_classifier/gru/while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/mul_5ì
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_4ReadVariableOp?gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype026
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_4Ç
9GRU_classifier/gru/while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_6/stackË
;GRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack_1Ë
;GRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack_2à
3GRU_classifier/gru/while/gru_cell_2/strided_slice_6StridedSlice<GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_4:value:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_6
,GRU_classifier/gru/while/gru_cell_2/MatMul_3MatMul-GRU_classifier/gru/while/gru_cell_2/mul_3:z:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,GRU_classifier/gru/while/gru_cell_2/MatMul_3ì
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_5ReadVariableOp?gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype026
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_5Ç
9GRU_classifier/gru/while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_7/stackË
;GRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack_1Ë
;GRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack_2à
3GRU_classifier/gru/while/gru_cell_2/strided_slice_7StridedSlice<GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_5:value:0BGRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_7
,GRU_classifier/gru/while/gru_cell_2/MatMul_4MatMul-GRU_classifier/gru/while/gru_cell_2/mul_4:z:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,GRU_classifier/gru/while/gru_cell_2/MatMul_4À
9GRU_classifier/gru/while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_8/stackÄ
;GRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack_1Ä
;GRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack_2Ä
3GRU_classifier/gru/while/gru_cell_2/strided_slice_8StridedSlice4GRU_classifier/gru/while/gru_cell_2/unstack:output:1BGRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_8
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_3BiasAdd6GRU_classifier/gru/while/gru_cell_2/MatMul_3:product:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_3À
9GRU_classifier/gru/while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell_2/strided_slice_9/stackÄ
;GRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack_1Ä
;GRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;GRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack_2²
3GRU_classifier/gru/while/gru_cell_2/strided_slice_9StridedSlice4GRU_classifier/gru/while/gru_cell_2/unstack:output:1BGRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack_1:output:0DGRU_classifier/gru/while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 25
3GRU_classifier/gru/while/gru_cell_2/strided_slice_9
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_4BiasAdd6GRU_classifier/gru/while/gru_cell_2/MatMul_4:product:0<GRU_classifier/gru/while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_4û
'GRU_classifier/gru/while/gru_cell_2/addAddV24GRU_classifier/gru/while/gru_cell_2/BiasAdd:output:06GRU_classifier/gru/while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell_2/addÄ
+GRU_classifier/gru/while/gru_cell_2/SigmoidSigmoid+GRU_classifier/gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell_2/Sigmoid
)GRU_classifier/gru/while/gru_cell_2/add_1AddV26GRU_classifier/gru/while/gru_cell_2/BiasAdd_1:output:06GRU_classifier/gru/while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/add_1Ê
-GRU_classifier/gru/while/gru_cell_2/Sigmoid_1Sigmoid-GRU_classifier/gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell_2/Sigmoid_1ì
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6ReadVariableOp?gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype026
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6É
:GRU_classifier/gru/while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2<
:GRU_classifier/gru/while/gru_cell_2/strided_slice_10/stackÍ
<GRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2>
<GRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack_1Í
<GRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<GRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack_2å
4GRU_classifier/gru/while/gru_cell_2/strided_slice_10StridedSlice<GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6:value:0CGRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack:output:0EGRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack_1:output:0EGRU_classifier/gru/while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4GRU_classifier/gru/while/gru_cell_2/strided_slice_10
,GRU_classifier/gru/while/gru_cell_2/MatMul_5MatMul-GRU_classifier/gru/while/gru_cell_2/mul_5:z:0=GRU_classifier/gru/while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,GRU_classifier/gru/while/gru_cell_2/MatMul_5Â
:GRU_classifier/gru/while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2<
:GRU_classifier/gru/while/gru_cell_2/strided_slice_11/stackÆ
<GRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<GRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack_1Æ
<GRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<GRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack_2Ç
4GRU_classifier/gru/while/gru_cell_2/strided_slice_11StridedSlice4GRU_classifier/gru/while/gru_cell_2/unstack:output:1CGRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack:output:0EGRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack_1:output:0EGRU_classifier/gru/while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask26
4GRU_classifier/gru/while/gru_cell_2/strided_slice_11
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_5BiasAdd6GRU_classifier/gru/while/gru_cell_2/MatMul_5:product:0=GRU_classifier/gru/while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell_2/BiasAdd_5ú
)GRU_classifier/gru/while/gru_cell_2/mul_6Mul1GRU_classifier/gru/while/gru_cell_2/Sigmoid_1:y:06GRU_classifier/gru/while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/mul_6ø
)GRU_classifier/gru/while/gru_cell_2/add_2AddV26GRU_classifier/gru/while/gru_cell_2/BiasAdd_2:output:0-GRU_classifier/gru/while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/add_2½
(GRU_classifier/gru/while/gru_cell_2/TanhTanh-GRU_classifier/gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(GRU_classifier/gru/while/gru_cell_2/Tanhè
)GRU_classifier/gru/while/gru_cell_2/mul_7Mul/GRU_classifier/gru/while/gru_cell_2/Sigmoid:y:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/mul_7
)GRU_classifier/gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)GRU_classifier/gru/while/gru_cell_2/sub/xð
'GRU_classifier/gru/while/gru_cell_2/subSub2GRU_classifier/gru/while/gru_cell_2/sub/x:output:0/GRU_classifier/gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell_2/subê
)GRU_classifier/gru/while/gru_cell_2/mul_8Mul+GRU_classifier/gru/while/gru_cell_2/sub:z:0,GRU_classifier/gru/while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/mul_8ï
)GRU_classifier/gru/while/gru_cell_2/add_3AddV2-GRU_classifier/gru/while/gru_cell_2/mul_7:z:0-GRU_classifier/gru/while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell_2/add_3£
'GRU_classifier/gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'GRU_classifier/gru/while/Tile/multiplesñ
GRU_classifier/gru/while/TileTileEGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem:item:00GRU_classifier/gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
GRU_classifier/gru/while/Tile
!GRU_classifier/gru/while/SelectV2SelectV2&GRU_classifier/gru/while/Tile:output:0-GRU_classifier/gru/while/gru_cell_2/add_3:z:0&gru_classifier_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/while/SelectV2§
)GRU_classifier/gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)GRU_classifier/gru/while/Tile_1/multiples÷
GRU_classifier/gru/while/Tile_1TileEGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem:item:02GRU_classifier/gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
GRU_classifier/gru/while/Tile_1
#GRU_classifier/gru/while/SelectV2_1SelectV2(GRU_classifier/gru/while/Tile_1:output:0-GRU_classifier/gru/while/gru_cell_2/add_3:z:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/while/SelectV2_1º
=GRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&gru_classifier_gru_while_placeholder_1$gru_classifier_gru_while_placeholder*GRU_classifier/gru/while/SelectV2:output:0*
_output_shapes
: *
element_dtype02?
=GRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItem
GRU_classifier/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
GRU_classifier/gru/while/add/yµ
GRU_classifier/gru/while/addAddV2$gru_classifier_gru_while_placeholder'GRU_classifier/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_classifier/gru/while/add
 GRU_classifier/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 GRU_classifier/gru/while/add_1/yÕ
GRU_classifier/gru/while/add_1AddV2>gru_classifier_gru_while_gru_classifier_gru_while_loop_counter)GRU_classifier/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
GRU_classifier/gru/while/add_1
!GRU_classifier/gru/while/IdentityIdentity"GRU_classifier/gru/while/add_1:z:03^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp5^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_15^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_25^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_35^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_45^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_55^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2#
!GRU_classifier/gru/while/Identity¼
#GRU_classifier/gru/while/Identity_1IdentityDgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations3^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp5^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_15^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_25^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_35^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_45^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_55^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_1
#GRU_classifier/gru/while/Identity_2Identity GRU_classifier/gru/while/add:z:03^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp5^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_15^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_25^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_35^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_45^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_55^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_2Å
#GRU_classifier/gru/while/Identity_3IdentityMGRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp5^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_15^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_25^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_35^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_45^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_55^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_3³
#GRU_classifier/gru/while/Identity_4Identity*GRU_classifier/gru/while/SelectV2:output:03^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp5^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_15^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_25^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_35^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_45^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_55^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/while/Identity_4µ
#GRU_classifier/gru/while/Identity_5Identity,GRU_classifier/gru/while/SelectV2_1:output:03^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp5^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_15^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_25^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_35^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_45^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_55^GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/while/Identity_5"
=gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource?gru_classifier_gru_while_gru_cell_2_readvariableop_1_resource_0"
=gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource?gru_classifier_gru_while_gru_cell_2_readvariableop_4_resource_0"|
;gru_classifier_gru_while_gru_cell_2_readvariableop_resource=gru_classifier_gru_while_gru_cell_2_readvariableop_resource_0"|
;gru_classifier_gru_while_gru_classifier_gru_strided_slice_1=gru_classifier_gru_while_gru_classifier_gru_strided_slice_1_0"O
!gru_classifier_gru_while_identity*GRU_classifier/gru/while/Identity:output:0"S
#gru_classifier_gru_while_identity_1,GRU_classifier/gru/while/Identity_1:output:0"S
#gru_classifier_gru_while_identity_2,GRU_classifier/gru/while/Identity_2:output:0"S
#gru_classifier_gru_while_identity_3,GRU_classifier/gru/while/Identity_3:output:0"S
#gru_classifier_gru_while_identity_4,GRU_classifier/gru/while/Identity_4:output:0"S
#gru_classifier_gru_while_identity_5,GRU_classifier/gru/while/Identity_5:output:0"ü
{gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0"ô
wgru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensorygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 2h
2GRU_classifier/gru/while/gru_cell_2/ReadVariableOp2GRU_classifier/gru/while/gru_cell_2/ReadVariableOp2l
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_14GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_12l
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_24GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_22l
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_34GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_32l
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_44GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_42l
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_54GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_52l
4GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_64GRU_classifier/gru/while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Á"
£
while_body_44107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_2_44129_0:`+
while_gru_cell_2_44131_0:	¬`*
while_gru_cell_2_44133_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_2_44129:`)
while_gru_cell_2_44131:	¬`(
while_gru_cell_2_44133: `¢(while/gru_cell_2/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¬
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_44129_0while_gru_cell_2_44131_0while_gru_cell_2_44133_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *N
fIRG
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_440942*
(while/gru_cell_2/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1)^while/gru_cell_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"2
while_gru_cell_2_44129while_gru_cell_2_44129_0"2
while_gru_cell_2_44131while_gru_cell_2_44131_0"2
while_gru_cell_2_44133while_gru_cell_2_44133_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2T
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ÿ·
æ
while_body_46786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`?
,while_gru_cell_2_readvariableop_1_resource_0:	¬`>
,while_gru_cell_2_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`=
*while_gru_cell_2_readvariableop_1_resource:	¬`<
*while_gru_cell_2_readvariableop_4_resource: `¢while/gru_cell_2/ReadVariableOp¢!while/gru_cell_2/ReadVariableOp_1¢!while/gru_cell_2/ReadVariableOp_2¢!while/gru_cell_2/ReadVariableOp_3¢!while/gru_cell_2/ReadVariableOp_4¢!while/gru_cell_2/ReadVariableOp_5¢!while/gru_cell_2/ReadVariableOp_6Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¤
 while/gru_cell_2/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell_2/ones_like/ConstÉ
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/ones_like
"while/gru_cell_2/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/gru_cell_2/ones_like_1/Shape
"while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/gru_cell_2/ones_like_1/ConstÐ
while/gru_cell_2/ones_like_1Fill+while/gru_cell_2/ones_like_1/Shape:output:0+while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/ones_like_1­
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02!
while/gru_cell_2/ReadVariableOp
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell_2/unstack½
while/gru_cell_2/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mulÁ
while/gru_cell_2/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_1Á
while/gru_cell_2/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_2´
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_1
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack¡
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice/stack_1¡
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2å
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice±
while/gru_cell_2/MatMulMatMulwhile/gru_cell_2/mul:z:0'while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul´
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_2¡
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_1/stack¥
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_1/stack_1¥
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2ï
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1¹
while/gru_cell_2/MatMul_1MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_1´
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_3¡
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell_2/strided_slice_2/stack¥
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1¥
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2ï
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2¹
while/gru_cell_2/MatMul_2MatMulwhile/gru_cell_2/mul_2:z:0)while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_2
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_3/stack_1
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2Ò
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_3¿
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_4/stack
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_4/stack_1
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2À
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_4Å
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_1
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell_2/strided_slice_5/stack
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2Ð
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2"
 while/gru_cell_2/strided_slice_5Å
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_2¥
while/gru_cell_2/mul_3Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_3¥
while/gru_cell_2/mul_4Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_4¥
while/gru_cell_2/mul_5Mulwhile_placeholder_2%while/gru_cell_2/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_5³
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_4¡
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack¥
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_6/stack_1¥
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2î
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6¹
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul_3:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_3³
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_5¡
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_7/stack¥
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_7/stack_1¥
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2î
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7¹
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_4:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_4
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_8/stack_1
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2Ò
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_8Å
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_3
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_9/stack
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_9/stack_1
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2À
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_9Å
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_4¯
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoidµ
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_1
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoid_1³
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_6£
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/gru_cell_2/strided_slice_10/stack§
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1§
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2ó
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10º
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_5:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_5
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2)
'while/gru_cell_2/strided_slice_11/stack 
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1 
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2Õ
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2#
!while/gru_cell_2/strided_slice_11Æ
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_5®
while/gru_cell_2/mul_6Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_6¬
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_2
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Tanh
while/gru_cell_2/mul_7Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_7u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_2/sub/x¤
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/sub
while/gru_cell_2/mul_8Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_8£
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_7:z:0while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ø
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityë
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ú
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ã'
¹
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45632

inputs
	gru_45607:`
	gru_45609:	¬`
	gru_45611: `
output_45614: 
output_45616:
identity¢gru/StatefulPartitionedCall¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallã
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_447912
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_45607	gru_45609	gru_45611*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_455712
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_45614output_45616*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_451252 
output/StatefulPartitionedCall½
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45609*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulÐ
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_45611*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall8^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Â 
ø
A__inference_output_layer_call_and_return_conditional_losses_45125

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd¥
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


E__inference_gru_cell_2_layer_call_and_return_conditional_losses_44094

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ones_like_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ÿ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ì
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2Ú
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2ê
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_2e
mul_3Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5~
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_3~
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2ì
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2Ú
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stack
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2ï
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3Í
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulà
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÞ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityâ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
9
¹

__inference__traced_save_48499
file_prefix,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_gru_gru_cell_2_kernel_read_readvariableop>
:savev2_gru_gru_cell_2_recurrent_kernel_read_readvariableop2
.savev2_gru_gru_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop;
7savev2_adam_gru_gru_cell_2_kernel_m_read_readvariableopE
Asavev2_adam_gru_gru_cell_2_recurrent_kernel_m_read_readvariableop9
5savev2_adam_gru_gru_cell_2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop;
7savev2_adam_gru_gru_cell_2_kernel_v_read_readvariableopE
Asavev2_adam_gru_gru_cell_2_recurrent_kernel_v_read_readvariableop9
5savev2_adam_gru_gru_cell_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesº
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÀ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_gru_gru_cell_2_kernel_read_readvariableop:savev2_gru_gru_cell_2_recurrent_kernel_read_readvariableop.savev2_gru_gru_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop7savev2_adam_gru_gru_cell_2_kernel_m_read_readvariableopAsavev2_adam_gru_gru_cell_2_recurrent_kernel_m_read_readvariableop5savev2_adam_gru_gru_cell_2_bias_m_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop7savev2_adam_gru_gru_cell_2_kernel_v_read_readvariableopAsavev2_adam_gru_gru_cell_2_recurrent_kernel_v_read_readvariableop5savev2_adam_gru_gru_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*¸
_input_shapes¦
£: : :: : : : : :	¬`: `:`: : : : : ::	¬`: `:`: ::	¬`: `:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	¬`:$	 

_output_shapes

: `:$
 

_output_shapes

:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	¬`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	¬`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:

_output_shapes
: 
ë	
^
B__inference_masking_layer_call_and_return_conditional_losses_44791

inputs
identity]

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2

NotEqual/y}
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Any/reduction_indices
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
mul
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
û
²
#__inference_gru_layer_call_fn_46644

inputs
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_450872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ÑT
õ
>__inference_gru_layer_call_and_return_conditional_losses_44183

inputs"
gru_cell_2_44095:`#
gru_cell_2_44097:	¬`"
gru_cell_2_44099: `
identity¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_2/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2ë
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_44095gru_cell_2_44097gru_cell_2_44099*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *N
fIRG
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_440942$
"gru_cell_2/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterß
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_44095gru_cell_2_44097gru_cell_2_44099*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44107*
condR
while_cond_44106*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÄ
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_2_44097*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mul×
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_2_44099*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mul
IdentityIdentitytranspose_1:y:08^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp#^gru_cell_2/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
å
ó
.__inference_GRU_classifier_layer_call_fn_45783

inputs
unknown:`
	unknown_0:	¬`
	unknown_1: `
	unknown_2: 
	unknown_3:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8 *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_456322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Á"
£
while_body_44439
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_2_44461_0:`+
while_gru_cell_2_44463_0:	¬`*
while_gru_cell_2_44465_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_2_44461:`)
while_gru_cell_2_44463:	¬`(
while_gru_cell_2_44465: `¢(while/gru_cell_2/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¬
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_44461_0while_gru_cell_2_44463_0while_gru_cell_2_44465_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *N
fIRG
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_443722*
(while/gru_cell_2/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1)^while/gru_cell_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"2
while_gru_cell_2_44461while_gru_cell_2_44461_0"2
while_gru_cell_2_44463while_gru_cell_2_44463_0"2
while_gru_cell_2_44465while_gru_cell_2_44465_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2T
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ë	
^
B__inference_masking_layer_call_and_return_conditional_losses_46599

inputs
identity]

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2

NotEqual/y}
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Any/reduction_indices
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
mul
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
þ»

E__inference_gru_cell_2_layer_call_and_return_conditional_losses_44372

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeÑ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Ï¨2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape×
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ãÒ´2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_1/GreaterEqual/yÇ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeÖ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2{2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_2/GreaterEqual/yÇ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_2/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ones_like_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/Const
dropout_3/MulMulones_like_1:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeÖ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2³´2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_3/GreaterEqual/yÆ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_3/Mul_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/ShapeÖ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2à2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_4/GreaterEqual/yÆ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/ShapeÖ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ñ²Ó2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
dropout_5/GreaterEqual/yÆ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_5/Mul_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ÿ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ì
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2Ú
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2ê
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5~
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_3~
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2ì
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2Ú
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stack
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2ï
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3Í
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype029
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpÉ
(gru/gru_cell_2/kernel/Regularizer/SquareSquare?gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2*
(gru/gru_cell_2/kernel/Regularizer/Square£
'gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'gru/gru_cell_2/kernel/Regularizer/ConstÖ
%gru/gru_cell_2/kernel/Regularizer/SumSum,gru/gru_cell_2/kernel/Regularizer/Square:y:00gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/Sum
'gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2)
'gru/gru_cell_2/kernel/Regularizer/mul/xØ
%gru/gru_cell_2/kernel/Regularizer/mulMul0gru/gru_cell_2/kernel/Regularizer/mul/x:output:0.gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%gru/gru_cell_2/kernel/Regularizer/mulà
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02C
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpæ
2gru/gru_cell_2/recurrent_kernel/Regularizer/SquareSquareIgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `24
2gru/gru_cell_2/recurrent_kernel/Regularizer/Square·
1gru/gru_cell_2/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       23
1gru/gru_cell_2/recurrent_kernel/Regularizer/Constþ
/gru/gru_cell_2/recurrent_kernel/Regularizer/SumSum6gru/gru_cell_2/recurrent_kernel/Regularizer/Square:y:0:gru/gru_cell_2/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/Sum«
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<23
1gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulMul:gru/gru_cell_2/recurrent_kernel/Regularizer/mul/x:output:08gru/gru_cell_2/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 21
/gru/gru_cell_2/recurrent_kernel/Regularizer/mulÞ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityâ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_68^gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpB^gru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62r
7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp7gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Agru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOpAgru/gru_cell_2/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
©
í
gru_while_body_46330$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0c
_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0@
.gru_while_gru_cell_2_readvariableop_resource_0:`C
0gru_while_gru_cell_2_readvariableop_1_resource_0:	¬`B
0gru_while_gru_cell_2_readvariableop_4_resource_0: `
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_identity_5!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensora
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor>
,gru_while_gru_cell_2_readvariableop_resource:`A
.gru_while_gru_cell_2_readvariableop_1_resource:	¬`@
.gru_while_gru_cell_2_readvariableop_4_resource: `¢#gru/while/gru_cell_2/ReadVariableOp¢%gru/while/gru_cell_2/ReadVariableOp_1¢%gru/while/gru_cell_2/ReadVariableOp_2¢%gru/while/gru_cell_2/ReadVariableOp_3¢%gru/while/gru_cell_2/ReadVariableOp_4¢%gru/while/gru_cell_2/ReadVariableOp_5¢%gru/while/gru_cell_2/ReadVariableOp_6Ë
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeì
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItemÏ
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeõ
/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0gru_while_placeholderFgru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
21
/gru/while/TensorArrayV2Read_1/TensorListGetItem°
$gru/while/gru_cell_2/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/ones_like/Shape
$gru/while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$gru/while/gru_cell_2/ones_like/ConstÙ
gru/while/gru_cell_2/ones_likeFill-gru/while/gru_cell_2/ones_like/Shape:output:0-gru/while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
gru/while/gru_cell_2/ones_like
"gru/while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell_2/dropout/ConstÔ
 gru/while/gru_cell_2/dropout/MulMul'gru/while/gru_cell_2/ones_like:output:0+gru/while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 gru/while/gru_cell_2/dropout/Mul
"gru/while/gru_cell_2/dropout/ShapeShape'gru/while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell_2/dropout/Shape
9gru/while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÍA2;
9gru/while/gru_cell_2/dropout/random_uniform/RandomUniform
+gru/while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2-
+gru/while/gru_cell_2/dropout/GreaterEqual/y
)gru/while/gru_cell_2/dropout/GreaterEqualGreaterEqualBgru/while/gru_cell_2/dropout/random_uniform/RandomUniform:output:04gru/while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2+
)gru/while/gru_cell_2/dropout/GreaterEqual¿
!gru/while/gru_cell_2/dropout/CastCast-gru/while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru/while/gru_cell_2/dropout/CastÏ
"gru/while/gru_cell_2/dropout/Mul_1Mul$gru/while/gru_cell_2/dropout/Mul:z:0%gru/while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"gru/while/gru_cell_2/dropout/Mul_1
$gru/while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru/while/gru_cell_2/dropout_1/ConstÚ
"gru/while/gru_cell_2/dropout_1/MulMul'gru/while/gru_cell_2/ones_like:output:0-gru/while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"gru/while/gru_cell_2/dropout_1/Mul£
$gru/while/gru_cell_2/dropout_1/ShapeShape'gru/while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/dropout_1/Shape
;gru/while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform-gru/while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ýÍå2=
;gru/while/gru_cell_2/dropout_1/random_uniform/RandomUniform£
-gru/while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2/
-gru/while/gru_cell_2/dropout_1/GreaterEqual/y
+gru/while/gru_cell_2/dropout_1/GreaterEqualGreaterEqualDgru/while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:06gru/while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2-
+gru/while/gru_cell_2/dropout_1/GreaterEqualÅ
#gru/while/gru_cell_2/dropout_1/CastCast/gru/while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#gru/while/gru_cell_2/dropout_1/Cast×
$gru/while/gru_cell_2/dropout_1/Mul_1Mul&gru/while/gru_cell_2/dropout_1/Mul:z:0'gru/while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$gru/while/gru_cell_2/dropout_1/Mul_1
$gru/while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru/while/gru_cell_2/dropout_2/ConstÚ
"gru/while/gru_cell_2/dropout_2/MulMul'gru/while/gru_cell_2/ones_like:output:0-gru/while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"gru/while/gru_cell_2/dropout_2/Mul£
$gru/while/gru_cell_2/dropout_2/ShapeShape'gru/while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/dropout_2/Shape
;gru/while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform-gru/while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ªæ2=
;gru/while/gru_cell_2/dropout_2/random_uniform/RandomUniform£
-gru/while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2/
-gru/while/gru_cell_2/dropout_2/GreaterEqual/y
+gru/while/gru_cell_2/dropout_2/GreaterEqualGreaterEqualDgru/while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:06gru/while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2-
+gru/while/gru_cell_2/dropout_2/GreaterEqualÅ
#gru/while/gru_cell_2/dropout_2/CastCast/gru/while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#gru/while/gru_cell_2/dropout_2/Cast×
$gru/while/gru_cell_2/dropout_2/Mul_1Mul&gru/while/gru_cell_2/dropout_2/Mul:z:0'gru/while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$gru/while/gru_cell_2/dropout_2/Mul_1
&gru/while/gru_cell_2/ones_like_1/ShapeShapegru_while_placeholder_3*
T0*
_output_shapes
:2(
&gru/while/gru_cell_2/ones_like_1/Shape
&gru/while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&gru/while/gru_cell_2/ones_like_1/Constà
 gru/while/gru_cell_2/ones_like_1Fill/gru/while/gru_cell_2/ones_like_1/Shape:output:0/gru/while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gru/while/gru_cell_2/ones_like_1
$gru/while/gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru/while/gru_cell_2/dropout_3/ConstÛ
"gru/while/gru_cell_2/dropout_3/MulMul)gru/while/gru_cell_2/ones_like_1:output:0-gru/while/gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gru/while/gru_cell_2/dropout_3/Mul¥
$gru/while/gru_cell_2/dropout_3/ShapeShape)gru/while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/dropout_3/Shape
;gru/while/gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform-gru/while/gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2¿Çà2=
;gru/while/gru_cell_2/dropout_3/random_uniform/RandomUniform£
-gru/while/gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2/
-gru/while/gru_cell_2/dropout_3/GreaterEqual/y
+gru/while/gru_cell_2/dropout_3/GreaterEqualGreaterEqualDgru/while/gru_cell_2/dropout_3/random_uniform/RandomUniform:output:06gru/while/gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+gru/while/gru_cell_2/dropout_3/GreaterEqualÄ
#gru/while/gru_cell_2/dropout_3/CastCast/gru/while/gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gru/while/gru_cell_2/dropout_3/CastÖ
$gru/while/gru_cell_2/dropout_3/Mul_1Mul&gru/while/gru_cell_2/dropout_3/Mul:z:0'gru/while/gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$gru/while/gru_cell_2/dropout_3/Mul_1
$gru/while/gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru/while/gru_cell_2/dropout_4/ConstÛ
"gru/while/gru_cell_2/dropout_4/MulMul)gru/while/gru_cell_2/ones_like_1:output:0-gru/while/gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gru/while/gru_cell_2/dropout_4/Mul¥
$gru/while/gru_cell_2/dropout_4/ShapeShape)gru/while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/dropout_4/Shape
;gru/while/gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform-gru/while/gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2·Ï2=
;gru/while/gru_cell_2/dropout_4/random_uniform/RandomUniform£
-gru/while/gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2/
-gru/while/gru_cell_2/dropout_4/GreaterEqual/y
+gru/while/gru_cell_2/dropout_4/GreaterEqualGreaterEqualDgru/while/gru_cell_2/dropout_4/random_uniform/RandomUniform:output:06gru/while/gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+gru/while/gru_cell_2/dropout_4/GreaterEqualÄ
#gru/while/gru_cell_2/dropout_4/CastCast/gru/while/gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gru/while/gru_cell_2/dropout_4/CastÖ
$gru/while/gru_cell_2/dropout_4/Mul_1Mul&gru/while/gru_cell_2/dropout_4/Mul:z:0'gru/while/gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$gru/while/gru_cell_2/dropout_4/Mul_1
$gru/while/gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru/while/gru_cell_2/dropout_5/ConstÛ
"gru/while/gru_cell_2/dropout_5/MulMul)gru/while/gru_cell_2/ones_like_1:output:0-gru/while/gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gru/while/gru_cell_2/dropout_5/Mul¥
$gru/while/gru_cell_2/dropout_5/ShapeShape)gru/while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2&
$gru/while/gru_cell_2/dropout_5/Shape
;gru/while/gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform-gru/while/gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2äK2=
;gru/while/gru_cell_2/dropout_5/random_uniform/RandomUniform£
-gru/while/gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2/
-gru/while/gru_cell_2/dropout_5/GreaterEqual/y
+gru/while/gru_cell_2/dropout_5/GreaterEqualGreaterEqualDgru/while/gru_cell_2/dropout_5/random_uniform/RandomUniform:output:06gru/while/gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+gru/while/gru_cell_2/dropout_5/GreaterEqualÄ
#gru/while/gru_cell_2/dropout_5/CastCast/gru/while/gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gru/while/gru_cell_2/dropout_5/CastÖ
$gru/while/gru_cell_2/dropout_5/Mul_1Mul&gru/while/gru_cell_2/dropout_5/Mul:z:0'gru/while/gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$gru/while/gru_cell_2/dropout_5/Mul_1¹
#gru/while/gru_cell_2/ReadVariableOpReadVariableOp.gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02%
#gru/while/gru_cell_2/ReadVariableOp©
gru/while/gru_cell_2/unstackUnpack+gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell_2/unstackÌ
gru/while/gru_cell_2/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell_2/mulÒ
gru/while/gru_cell_2/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0(gru/while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell_2/mul_1Ò
gru/while/gru_cell_2/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0(gru/while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell_2/mul_2À
%gru/while/gru_cell_2/ReadVariableOp_1ReadVariableOp0gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_1¥
(gru/while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell_2/strided_slice/stack©
*gru/while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice/stack_1©
*gru/while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell_2/strided_slice/stack_2ý
"gru/while/gru_cell_2/strided_sliceStridedSlice-gru/while/gru_cell_2/ReadVariableOp_1:value:01gru/while/gru_cell_2/strided_slice/stack:output:03gru/while/gru_cell_2/strided_slice/stack_1:output:03gru/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2$
"gru/while/gru_cell_2/strided_sliceÁ
gru/while/gru_cell_2/MatMulMatMulgru/while/gru_cell_2/mul:z:0+gru/while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMulÀ
%gru/while/gru_cell_2/ReadVariableOp_2ReadVariableOp0gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_2©
*gru/while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice_1/stack­
,gru/while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,gru/while/gru_cell_2/strided_slice_1/stack_1­
,gru/while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_1/stack_2
$gru/while/gru_cell_2/strided_slice_1StridedSlice-gru/while/gru_cell_2/ReadVariableOp_2:value:03gru/while/gru_cell_2/strided_slice_1/stack:output:05gru/while/gru_cell_2/strided_slice_1/stack_1:output:05gru/while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_1É
gru/while/gru_cell_2/MatMul_1MatMulgru/while/gru_cell_2/mul_1:z:0-gru/while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_1À
%gru/while/gru_cell_2/ReadVariableOp_3ReadVariableOp0gru_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_3©
*gru/while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell_2/strided_slice_2/stack­
,gru/while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,gru/while/gru_cell_2/strided_slice_2/stack_1­
,gru/while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_2/stack_2
$gru/while/gru_cell_2/strided_slice_2StridedSlice-gru/while/gru_cell_2/ReadVariableOp_3:value:03gru/while/gru_cell_2/strided_slice_2/stack:output:05gru/while/gru_cell_2/strided_slice_2/stack_1:output:05gru/while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_2É
gru/while/gru_cell_2/MatMul_2MatMulgru/while/gru_cell_2/mul_2:z:0-gru/while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_2¢
*gru/while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_3/stack¦
,gru/while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,gru/while/gru_cell_2/strided_slice_3/stack_1¦
,gru/while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_3/stack_2ê
$gru/while/gru_cell_2/strided_slice_3StridedSlice%gru/while/gru_cell_2/unstack:output:03gru/while/gru_cell_2/strided_slice_3/stack:output:05gru/while/gru_cell_2/strided_slice_3/stack_1:output:05gru/while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2&
$gru/while/gru_cell_2/strided_slice_3Ï
gru/while/gru_cell_2/BiasAddBiasAdd%gru/while/gru_cell_2/MatMul:product:0-gru/while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/BiasAdd¢
*gru/while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_4/stack¦
,gru/while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2.
,gru/while/gru_cell_2/strided_slice_4/stack_1¦
,gru/while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_4/stack_2Ø
$gru/while/gru_cell_2/strided_slice_4StridedSlice%gru/while/gru_cell_2/unstack:output:03gru/while/gru_cell_2/strided_slice_4/stack:output:05gru/while/gru_cell_2/strided_slice_4/stack_1:output:05gru/while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2&
$gru/while/gru_cell_2/strided_slice_4Õ
gru/while/gru_cell_2/BiasAdd_1BiasAdd'gru/while/gru_cell_2/MatMul_1:product:0-gru/while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_1¢
*gru/while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell_2/strided_slice_5/stack¦
,gru/while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,gru/while/gru_cell_2/strided_slice_5/stack_1¦
,gru/while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_5/stack_2è
$gru/while/gru_cell_2/strided_slice_5StridedSlice%gru/while/gru_cell_2/unstack:output:03gru/while/gru_cell_2/strided_slice_5/stack:output:05gru/while/gru_cell_2/strided_slice_5/stack_1:output:05gru/while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2&
$gru/while/gru_cell_2/strided_slice_5Õ
gru/while/gru_cell_2/BiasAdd_2BiasAdd'gru/while/gru_cell_2/MatMul_2:product:0-gru/while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_2´
gru/while/gru_cell_2/mul_3Mulgru_while_placeholder_3(gru/while/gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_3´
gru/while/gru_cell_2/mul_4Mulgru_while_placeholder_3(gru/while/gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_4´
gru/while/gru_cell_2/mul_5Mulgru_while_placeholder_3(gru/while/gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_5¿
%gru/while/gru_cell_2/ReadVariableOp_4ReadVariableOp0gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_4©
*gru/while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice_6/stack­
,gru/while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,gru/while/gru_cell_2/strided_slice_6/stack_1­
,gru/while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_6/stack_2
$gru/while/gru_cell_2/strided_slice_6StridedSlice-gru/while/gru_cell_2/ReadVariableOp_4:value:03gru/while/gru_cell_2/strided_slice_6/stack:output:05gru/while/gru_cell_2/strided_slice_6/stack_1:output:05gru/while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_6É
gru/while/gru_cell_2/MatMul_3MatMulgru/while/gru_cell_2/mul_3:z:0-gru/while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_3¿
%gru/while/gru_cell_2/ReadVariableOp_5ReadVariableOp0gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_5©
*gru/while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell_2/strided_slice_7/stack­
,gru/while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2.
,gru/while/gru_cell_2/strided_slice_7/stack_1­
,gru/while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru/while/gru_cell_2/strided_slice_7/stack_2
$gru/while/gru_cell_2/strided_slice_7StridedSlice-gru/while/gru_cell_2/ReadVariableOp_5:value:03gru/while/gru_cell_2/strided_slice_7/stack:output:05gru/while/gru_cell_2/strided_slice_7/stack_1:output:05gru/while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2&
$gru/while/gru_cell_2/strided_slice_7É
gru/while/gru_cell_2/MatMul_4MatMulgru/while/gru_cell_2/mul_4:z:0-gru/while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_4¢
*gru/while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_8/stack¦
,gru/while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,gru/while/gru_cell_2/strided_slice_8/stack_1¦
,gru/while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_8/stack_2ê
$gru/while/gru_cell_2/strided_slice_8StridedSlice%gru/while/gru_cell_2/unstack:output:13gru/while/gru_cell_2/strided_slice_8/stack:output:05gru/while/gru_cell_2/strided_slice_8/stack_1:output:05gru/while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2&
$gru/while/gru_cell_2/strided_slice_8Õ
gru/while/gru_cell_2/BiasAdd_3BiasAdd'gru/while/gru_cell_2/MatMul_3:product:0-gru/while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_3¢
*gru/while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell_2/strided_slice_9/stack¦
,gru/while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2.
,gru/while/gru_cell_2/strided_slice_9/stack_1¦
,gru/while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gru/while/gru_cell_2/strided_slice_9/stack_2Ø
$gru/while/gru_cell_2/strided_slice_9StridedSlice%gru/while/gru_cell_2/unstack:output:13gru/while/gru_cell_2/strided_slice_9/stack:output:05gru/while/gru_cell_2/strided_slice_9/stack_1:output:05gru/while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2&
$gru/while/gru_cell_2/strided_slice_9Õ
gru/while/gru_cell_2/BiasAdd_4BiasAdd'gru/while/gru_cell_2/MatMul_4:product:0-gru/while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_4¿
gru/while/gru_cell_2/addAddV2%gru/while/gru_cell_2/BiasAdd:output:0'gru/while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add
gru/while/gru_cell_2/SigmoidSigmoidgru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/SigmoidÅ
gru/while/gru_cell_2/add_1AddV2'gru/while/gru_cell_2/BiasAdd_1:output:0'gru/while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add_1
gru/while/gru_cell_2/Sigmoid_1Sigmoidgru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/Sigmoid_1¿
%gru/while/gru_cell_2/ReadVariableOp_6ReadVariableOp0gru_while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02'
%gru/while/gru_cell_2/ReadVariableOp_6«
+gru/while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2-
+gru/while/gru_cell_2/strided_slice_10/stack¯
-gru/while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-gru/while/gru_cell_2/strided_slice_10/stack_1¯
-gru/while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-gru/while/gru_cell_2/strided_slice_10/stack_2
%gru/while/gru_cell_2/strided_slice_10StridedSlice-gru/while/gru_cell_2/ReadVariableOp_6:value:04gru/while/gru_cell_2/strided_slice_10/stack:output:06gru/while/gru_cell_2/strided_slice_10/stack_1:output:06gru/while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2'
%gru/while/gru_cell_2/strided_slice_10Ê
gru/while/gru_cell_2/MatMul_5MatMulgru/while/gru_cell_2/mul_5:z:0.gru/while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/MatMul_5¤
+gru/while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2-
+gru/while/gru_cell_2/strided_slice_11/stack¨
-gru/while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-gru/while/gru_cell_2/strided_slice_11/stack_1¨
-gru/while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gru/while/gru_cell_2/strided_slice_11/stack_2í
%gru/while/gru_cell_2/strided_slice_11StridedSlice%gru/while/gru_cell_2/unstack:output:14gru/while/gru_cell_2/strided_slice_11/stack:output:06gru/while/gru_cell_2/strided_slice_11/stack_1:output:06gru/while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2'
%gru/while/gru_cell_2/strided_slice_11Ö
gru/while/gru_cell_2/BiasAdd_5BiasAdd'gru/while/gru_cell_2/MatMul_5:product:0.gru/while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell_2/BiasAdd_5¾
gru/while/gru_cell_2/mul_6Mul"gru/while/gru_cell_2/Sigmoid_1:y:0'gru/while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_6¼
gru/while/gru_cell_2/add_2AddV2'gru/while/gru_cell_2/BiasAdd_2:output:0gru/while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add_2
gru/while/gru_cell_2/TanhTanhgru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/Tanh¬
gru/while/gru_cell_2/mul_7Mul gru/while/gru_cell_2/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_7}
gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell_2/sub/x´
gru/while/gru_cell_2/subSub#gru/while/gru_cell_2/sub/x:output:0 gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/sub®
gru/while/gru_cell_2/mul_8Mulgru/while/gru_cell_2/sub:z:0gru/while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/mul_8³
gru/while/gru_cell_2/add_3AddV2gru/while/gru_cell_2/mul_7:z:0gru/while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell_2/add_3
gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile/multiplesµ
gru/while/TileTile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0!gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/Tile¸
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell_2/add_3:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/SelectV2
gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile_1/multiples»
gru/while/Tile_1Tile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0#gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/Tile_1¾
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell_2/add_3:z:0gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/SelectV2_1ï
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/SelectV2:output:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1
gru/while/IdentityIdentitygru/while/add_1:z:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1
gru/while/Identity_2Identitygru/while/add:z:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2¯
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/SelectV2:output:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_4
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0$^gru/while/gru_cell_2/ReadVariableOp&^gru/while/gru_cell_2/ReadVariableOp_1&^gru/while/gru_cell_2/ReadVariableOp_2&^gru/while/gru_cell_2/ReadVariableOp_3&^gru/while/gru_cell_2/ReadVariableOp_4&^gru/while/gru_cell_2/ReadVariableOp_5&^gru/while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_5"b
.gru_while_gru_cell_2_readvariableop_1_resource0gru_while_gru_cell_2_readvariableop_1_resource_0"b
.gru_while_gru_cell_2_readvariableop_4_resource0gru_while_gru_cell_2_readvariableop_4_resource_0"^
,gru_while_gru_cell_2_readvariableop_resource.gru_while_gru_cell_2_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"5
gru_while_identity_5gru/while/Identity_5:output:0"À
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 2J
#gru/while/gru_cell_2/ReadVariableOp#gru/while/gru_cell_2/ReadVariableOp2N
%gru/while/gru_cell_2/ReadVariableOp_1%gru/while/gru_cell_2/ReadVariableOp_12N
%gru/while/gru_cell_2/ReadVariableOp_2%gru/while/gru_cell_2/ReadVariableOp_22N
%gru/while/gru_cell_2/ReadVariableOp_3%gru/while/gru_cell_2/ReadVariableOp_32N
%gru/while/gru_cell_2/ReadVariableOp_4%gru/while/gru_cell_2/ReadVariableOp_42N
%gru/while/gru_cell_2/ReadVariableOp_5%gru/while/gru_cell_2/ReadVariableOp_52N
%gru/while/gru_cell_2/ReadVariableOp_6%gru/while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
÷
æ
while_body_47815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`?
,while_gru_cell_2_readvariableop_1_resource_0:	¬`>
,while_gru_cell_2_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`=
*while_gru_cell_2_readvariableop_1_resource:	¬`<
*while_gru_cell_2_readvariableop_4_resource: `¢while/gru_cell_2/ReadVariableOp¢!while/gru_cell_2/ReadVariableOp_1¢!while/gru_cell_2/ReadVariableOp_2¢!while/gru_cell_2/ReadVariableOp_3¢!while/gru_cell_2/ReadVariableOp_4¢!while/gru_cell_2/ReadVariableOp_5¢!while/gru_cell_2/ReadVariableOp_6Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¤
 while/gru_cell_2/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell_2/ones_like/ConstÉ
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/ones_like
while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_2/dropout/ConstÄ
while/gru_cell_2/dropout/MulMul#while/gru_cell_2/ones_like:output:0'while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/dropout/Mul
while/gru_cell_2/dropout/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_2/dropout/Shape
5while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÔÆÁ27
5while/gru_cell_2/dropout/random_uniform/RandomUniform
'while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell_2/dropout/GreaterEqual/y
%while/gru_cell_2/dropout/GreaterEqualGreaterEqual>while/gru_cell_2/dropout/random_uniform/RandomUniform:output:00while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell_2/dropout/GreaterEqual³
while/gru_cell_2/dropout/CastCast)while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/dropout/Cast¿
while/gru_cell_2/dropout/Mul_1Mul while/gru_cell_2/dropout/Mul:z:0!while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout/Mul_1
 while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_1/ConstÊ
while/gru_cell_2/dropout_1/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout_1/Mul
 while/gru_cell_2/dropout_1/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_1/Shape
7while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ï²29
7while/gru_cell_2/dropout_1/random_uniform/RandomUniform
)while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_1/GreaterEqual/y
'while/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'while/gru_cell_2/dropout_1/GreaterEqual¹
while/gru_cell_2/dropout_1/CastCast+while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
while/gru_cell_2/dropout_1/CastÇ
 while/gru_cell_2/dropout_1/Mul_1Mul"while/gru_cell_2/dropout_1/Mul:z:0#while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 while/gru_cell_2/dropout_1/Mul_1
 while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_2/ConstÊ
while/gru_cell_2/dropout_2/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell_2/dropout_2/Mul
 while/gru_cell_2/dropout_2/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_2/Shape
7while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÈÁ29
7while/gru_cell_2/dropout_2/random_uniform/RandomUniform
)while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_2/GreaterEqual/y
'while/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'while/gru_cell_2/dropout_2/GreaterEqual¹
while/gru_cell_2/dropout_2/CastCast+while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
while/gru_cell_2/dropout_2/CastÇ
 while/gru_cell_2/dropout_2/Mul_1Mul"while/gru_cell_2/dropout_2/Mul:z:0#while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 while/gru_cell_2/dropout_2/Mul_1
"while/gru_cell_2/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2$
"while/gru_cell_2/ones_like_1/Shape
"while/gru_cell_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/gru_cell_2/ones_like_1/ConstÐ
while/gru_cell_2/ones_like_1Fill+while/gru_cell_2/ones_like_1/Shape:output:0+while/gru_cell_2/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/ones_like_1
 while/gru_cell_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_3/ConstË
while/gru_cell_2/dropout_3/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_3/Mul
 while/gru_cell_2/dropout_3/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_3/Shape
7while/gru_cell_2/dropout_3/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2­»29
7while/gru_cell_2/dropout_3/random_uniform/RandomUniform
)while/gru_cell_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_3/GreaterEqual/y
'while/gru_cell_2/dropout_3/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_3/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_3/GreaterEqual¸
while/gru_cell_2/dropout_3/CastCast+while/gru_cell_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_3/CastÆ
 while/gru_cell_2/dropout_3/Mul_1Mul"while/gru_cell_2/dropout_3/Mul:z:0#while/gru_cell_2/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_3/Mul_1
 while/gru_cell_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_4/ConstË
while/gru_cell_2/dropout_4/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_4/Mul
 while/gru_cell_2/dropout_4/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_4/Shape
7while/gru_cell_2/dropout_4/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ú¥)29
7while/gru_cell_2/dropout_4/random_uniform/RandomUniform
)while/gru_cell_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_4/GreaterEqual/y
'while/gru_cell_2/dropout_4/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_4/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_4/GreaterEqual¸
while/gru_cell_2/dropout_4/CastCast+while/gru_cell_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_4/CastÆ
 while/gru_cell_2/dropout_4/Mul_1Mul"while/gru_cell_2/dropout_4/Mul:z:0#while/gru_cell_2/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_4/Mul_1
 while/gru_cell_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_5/ConstË
while/gru_cell_2/dropout_5/MulMul%while/gru_cell_2/ones_like_1:output:0)while/gru_cell_2/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell_2/dropout_5/Mul
 while/gru_cell_2/dropout_5/ShapeShape%while/gru_cell_2/ones_like_1:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_5/Shape
7while/gru_cell_2/dropout_5/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ø¯29
7while/gru_cell_2/dropout_5/random_uniform/RandomUniform
)while/gru_cell_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)while/gru_cell_2/dropout_5/GreaterEqual/y
'while/gru_cell_2/dropout_5/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_5/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'while/gru_cell_2/dropout_5/GreaterEqual¸
while/gru_cell_2/dropout_5/CastCast+while/gru_cell_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/gru_cell_2/dropout_5/CastÆ
 while/gru_cell_2/dropout_5/Mul_1Mul"while/gru_cell_2/dropout_5/Mul:z:0#while/gru_cell_2/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/gru_cell_2/dropout_5/Mul_1­
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype02!
while/gru_cell_2/ReadVariableOp
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell_2/unstack¼
while/gru_cell_2/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mulÂ
while/gru_cell_2/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_1Â
while/gru_cell_2/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell_2/mul_2´
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_1
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack¡
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice/stack_1¡
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2å
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice±
while/gru_cell_2/MatMulMatMulwhile/gru_cell_2/mul:z:0'while/gru_cell_2/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul´
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_2¡
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_1/stack¥
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_1/stack_1¥
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2ï
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1¹
while/gru_cell_2/MatMul_1MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_1´
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02#
!while/gru_cell_2/ReadVariableOp_3¡
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell_2/strided_slice_2/stack¥
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1¥
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2ï
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2¹
while/gru_cell_2/MatMul_2MatMulwhile/gru_cell_2/mul_2:z:0)while/gru_cell_2/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_2
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_3/stack_1
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2Ò
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_3¿
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_4/stack
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_4/stack_1
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2À
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_4Å
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_1
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell_2/strided_slice_5/stack
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2Ð
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2"
 while/gru_cell_2/strided_slice_5Å
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_2¤
while/gru_cell_2/mul_3Mulwhile_placeholder_2$while/gru_cell_2/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_3¤
while/gru_cell_2/mul_4Mulwhile_placeholder_2$while/gru_cell_2/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_4¤
while/gru_cell_2/mul_5Mulwhile_placeholder_2$while/gru_cell_2/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_5³
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_4¡
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack¥
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_6/stack_1¥
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2î
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6¹
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul_3:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_3³
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_5¡
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_7/stack¥
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/gru_cell_2/strided_slice_7/stack_1¥
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2î
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7¹
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_4:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_4
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_8/stack_1
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2Ò
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2"
 while/gru_cell_2/strided_slice_8Å
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_3
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_9/stack
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2*
(while/gru_cell_2/strided_slice_9/stack_1
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2À
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2"
 while/gru_cell_2/strided_slice_9Å
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_4¯
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoidµ
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_1
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Sigmoid_1³
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02#
!while/gru_cell_2/ReadVariableOp_6£
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/gru_cell_2/strided_slice_10/stack§
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1§
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2ó
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10º
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_5:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/MatMul_5
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2)
'while/gru_cell_2/strided_slice_11/stack 
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1 
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2Õ
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2#
!while/gru_cell_2/strided_slice_11Æ
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/BiasAdd_5®
while/gru_cell_2/mul_6Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_6¬
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_2
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/Tanh
while/gru_cell_2/mul_7Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_7u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_2/sub/x¤
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/sub
while/gru_cell_2/mul_8Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/mul_8£
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_7:z:0while/gru_cell_2/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell_2/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ø
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityë
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ú
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
°

×
*__inference_gru_cell_2_layer_call_fn_48106

inputs
states_0
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity

identity_1¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *N
fIRG
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_443722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0

´
#__inference_gru_layer_call_fn_46633
inputs_0
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_445152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0


#GRU_classifier_gru_while_cond_43751B
>gru_classifier_gru_while_gru_classifier_gru_while_loop_counterH
Dgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations(
$gru_classifier_gru_while_placeholder*
&gru_classifier_gru_while_placeholder_1*
&gru_classifier_gru_while_placeholder_2*
&gru_classifier_gru_while_placeholder_3D
@gru_classifier_gru_while_less_gru_classifier_gru_strided_slice_1Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_43751___redundant_placeholder0Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_43751___redundant_placeholder1Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_43751___redundant_placeholder2Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_43751___redundant_placeholder3Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_43751___redundant_placeholder4%
!gru_classifier_gru_while_identity
Ï
GRU_classifier/gru/while/LessLess$gru_classifier_gru_while_placeholder@gru_classifier_gru_while_less_gru_classifier_gru_strided_slice_1*
T0*
_output_shapes
: 2
GRU_classifier/gru/while/Less
!GRU_classifier/gru/while/IdentityIdentity!GRU_classifier/gru/while/Less:z:0*
T0
*
_output_shapes
: 2#
!GRU_classifier/gru/while/Identity"O
!gru_classifier_gru_while_identity*GRU_classifier/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
E
input<
serving_default_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬G
output=
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¡É
ø0
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
V_default_save_signature
W__call__
*X&call_and_return_all_conditional_losses"Å.
_tf_keras_network©.{"name": "GRU_classifier", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "GRU_classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "name": "masking", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["masking", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["gru", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 300]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "GRU_classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "name": "masking", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["masking", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["gru", 0, 0, {}]]], "shared_object_id": 10}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 13}, "metrics": [[{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 14}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00039999998989515007, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
õ"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}

regularization_losses
trainable_variables
	variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"name": "masking", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "stateful": false, "must_restore_from_config": false, "class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}
ò
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"É
_tf_keras_rnn_layer«{"name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "inbound_nodes": [[["masking", 0, 0, {}]]], "shared_object_id": 7, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}}
û

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["gru", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 32]}}
­
iter

beta_1

beta_2
	decay
learning_ratemLmM mN!mO"mPvQvR vS!vT"vU"
	optimizer
 "
trackable_list_wrapper
C
 0
!1
"2
3
4"
trackable_list_wrapper
C
 0
!1
"2
3
4"
trackable_list_wrapper
Ê
#non_trainable_variables
$metrics
regularization_losses
trainable_variables

%layers
&layer_regularization_losses
'layer_metrics
	variables
W__call__
V_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
_serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
(metrics
)non_trainable_variables
regularization_losses
trainable_variables

*layers
+layer_regularization_losses
,layer_metrics
	variables
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object



 kernel
!recurrent_kernel
"bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
`__call__
*a&call_and_return_all_conditional_losses"ß
_tf_keras_layerÅ{"name": "gru_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 6}
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
¹
1non_trainable_variables
2metrics
regularization_losses
trainable_variables

3layers
4layer_regularization_losses

5states
6layer_metrics
	variables
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
: 2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
7metrics
8non_trainable_variables
regularization_losses
trainable_variables

9layers
:layer_regularization_losses
;layer_metrics
	variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(:&	¬`2gru/gru_cell_2/kernel
1:/ `2gru/gru_cell_2/recurrent_kernel
%:#`2gru/gru_cell_2/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
­
>metrics
?non_trainable_variables
-regularization_losses
.trainable_variables

@layers
Alayer_regularization_losses
Blayer_metrics
/	variables
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ô
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 17}
§
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"à
_tf_keras_metricÅ{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 14}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
$:" 2Adam/output/kernel/m
:2Adam/output/bias/m
-:+	¬`2Adam/gru/gru_cell_2/kernel/m
6:4 `2&Adam/gru/gru_cell_2/recurrent_kernel/m
*:(`2Adam/gru/gru_cell_2/bias/m
$:" 2Adam/output/kernel/v
:2Adam/output/bias/v
-:+	¬`2Adam/gru/gru_cell_2/kernel/v
6:4 `2&Adam/gru/gru_cell_2/recurrent_kernel/v
*:(`2Adam/gru/gru_cell_2/bias/v
ê2ç
 __inference__wrapped_model_43945Â
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *2¢/
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
2
.__inference_GRU_classifier_layer_call_fn_45157
.__inference_GRU_classifier_layer_call_fn_45768
.__inference_GRU_classifier_layer_call_fn_45783
.__inference_GRU_classifier_layer_call_fn_45660À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_46135
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_46583
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45689
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45718À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_masking_layer_call_fn_46588¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_masking_layer_call_and_return_conditional_losses_46599¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
#__inference_gru_layer_call_fn_46622
#__inference_gru_layer_call_fn_46633
#__inference_gru_layer_call_fn_46644
#__inference_gru_layer_call_fn_46655Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Û2Ø
>__inference_gru_layer_call_and_return_conditional_losses_46950
>__inference_gru_layer_call_and_return_conditional_losses_47341
>__inference_gru_layer_call_and_return_conditional_losses_47636
>__inference_gru_layer_call_and_return_conditional_losses_48027Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
&__inference_output_layer_call_fn_48036¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_output_layer_call_and_return_conditional_losses_48066¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÈBÅ
#__inference_signature_wrapper_45753input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_gru_cell_2_layer_call_fn_48092
*__inference_gru_cell_2_layer_call_fn_48106¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_48220
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_48382¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
__inference_loss_fn_0_48393
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
²2¯
__inference_loss_fn_1_48404
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ Ï
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45689" !D¢A
:¢7
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_45718" !D¢A
:¢7
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_46135" !E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_46583" !E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¦
.__inference_GRU_classifier_layer_call_fn_45157t" !D¢A
:¢7
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
.__inference_GRU_classifier_layer_call_fn_45660t" !D¢A
:¢7
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
.__inference_GRU_classifier_layer_call_fn_45768u" !E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
.__inference_GRU_classifier_layer_call_fn_45783u" !E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
 __inference__wrapped_model_43945" !<¢9
2¢/
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
ª "<ª9
7
output-*
outputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_48220¸" !]¢Z
S¢P
!
inputsÿÿÿÿÿÿÿÿÿ¬
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ 
$!

0/1/0ÿÿÿÿÿÿÿÿÿ 
 
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_48382¸" !]¢Z
S¢P
!
inputsÿÿÿÿÿÿÿÿÿ¬
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ 
$!

0/1/0ÿÿÿÿÿÿÿÿÿ 
 Ù
*__inference_gru_cell_2_layer_call_fn_48092ª" !]¢Z
S¢P
!
inputsÿÿÿÿÿÿÿÿÿ¬
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ 
"

1/0ÿÿÿÿÿÿÿÿÿ Ù
*__inference_gru_cell_2_layer_call_fn_48106ª" !]¢Z
S¢P
!
inputsÿÿÿÿÿÿÿÿÿ¬
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ 
"

1/0ÿÿÿÿÿÿÿÿÿ Î
>__inference_gru_layer_call_and_return_conditional_losses_46950" !P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Î
>__inference_gru_layer_call_and_return_conditional_losses_47341" !P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ç
>__inference_gru_layer_call_and_return_conditional_losses_47636" !I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ç
>__inference_gru_layer_call_and_return_conditional_losses_48027" !I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¥
#__inference_gru_layer_call_fn_46622~" !P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¥
#__inference_gru_layer_call_fn_46633~" !P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
#__inference_gru_layer_call_fn_46644w" !I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
#__inference_gru_layer_call_fn_46655w" !I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :
__inference_loss_fn_0_48393 ¢

¢ 
ª " :
__inference_loss_fn_1_48404!¢

¢ 
ª " º
B__inference_masking_layer_call_and_return_conditional_losses_46599t=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
'__inference_masking_layer_call_fn_46588g=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬»
A__inference_output_layer_call_and_return_conditional_losses_48066v<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
&__inference_output_layer_call_fn_48036i<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
#__inference_signature_wrapper_45753" !E¢B
¢ 
;ª8
6
input-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬"<ª9
7
output-*
outputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ß1
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
Ttype"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¦ª/
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

gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬`*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	¬`*
dtype0

gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*.
shared_namegru/gru_cell/recurrent_kernel

1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

: `*
dtype0
~
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*"
shared_namegru/gru_cell/bias
w
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
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

Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬`*+
shared_nameAdam/gru/gru_cell/kernel/m

.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes
:	¬`*
dtype0
¤
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m

8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/m

,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
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

Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬`*+
shared_nameAdam/gru/gru_cell/kernel/v

.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes
:	¬`*
dtype0
¤
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v

8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/v

,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes

:`*
dtype0

NoOpNoOp
%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Í$
valueÃ$BÀ$ B¹$
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
YW
VARIABLE_VALUEgru/gru_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEgru/gru_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEgru/gru_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
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
|z
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/gru/gru_cell/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/gru/gru_cell/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_inputPlaceholder*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
dtype0**
shape!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
®
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kerneloutput/kerneloutput/bias*
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
#__inference_signature_wrapper_14657
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ë	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp,Adam/gru/gru_cell/bias/m/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpConst*%
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
__inference__traced_save_17403

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcounttotal_1count_1Adam/output/kernel/mAdam/output/bias/mAdam/gru/gru_cell/kernel/m$Adam/gru/gru_cell/recurrent_kernel/mAdam/gru/gru_cell/bias/mAdam/output/kernel/vAdam/output/bias/vAdam/gru/gru_cell/kernel/v$Adam/gru/gru_cell/recurrent_kernel/vAdam/gru/gru_cell/bias/v*$
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
!__inference__traced_restore_17485ñÁ.
ë	
^
B__inference_masking_layer_call_and_return_conditional_losses_15503

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
¡
Ó
gru_while_body_15234$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0c
_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0>
,gru_while_gru_cell_readvariableop_resource_0:`A
.gru_while_gru_cell_readvariableop_1_resource_0:	¬`@
.gru_while_gru_cell_readvariableop_4_resource_0: `
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_identity_5!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensora
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor<
*gru_while_gru_cell_readvariableop_resource:`?
,gru_while_gru_cell_readvariableop_1_resource:	¬`>
,gru_while_gru_cell_readvariableop_4_resource: `¢!gru/while/gru_cell/ReadVariableOp¢#gru/while/gru_cell/ReadVariableOp_1¢#gru/while/gru_cell/ReadVariableOp_2¢#gru/while/gru_cell/ReadVariableOp_3¢#gru/while/gru_cell/ReadVariableOp_4¢#gru/while/gru_cell/ReadVariableOp_5¢#gru/while/gru_cell/ReadVariableOp_6Ë
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
/gru/while/TensorArrayV2Read_1/TensorListGetItem¬
"gru/while/gru_cell/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/Shape
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"gru/while/gru_cell/ones_like/ConstÑ
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/ones_like
 gru/while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru/while/gru_cell/dropout/ConstÌ
gru/while/gru_cell/dropout/MulMul%gru/while/gru_cell/ones_like:output:0)gru/while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
gru/while/gru_cell/dropout/Mul
 gru/while/gru_cell/dropout/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 gru/while/gru_cell/dropout/Shape
7gru/while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform)gru/while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ì±29
7gru/while/gru_cell/dropout/random_uniform/RandomUniform
)gru/while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2+
)gru/while/gru_cell/dropout/GreaterEqual/y
'gru/while/gru_cell/dropout/GreaterEqualGreaterEqual@gru/while/gru_cell/dropout/random_uniform/RandomUniform:output:02gru/while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'gru/while/gru_cell/dropout/GreaterEqual¹
gru/while/gru_cell/dropout/CastCast+gru/while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru/while/gru_cell/dropout/CastÇ
 gru/while/gru_cell/dropout/Mul_1Mul"gru/while/gru_cell/dropout/Mul:z:0#gru/while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 gru/while/gru_cell/dropout/Mul_1
"gru/while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_1/ConstÒ
 gru/while/gru_cell/dropout_1/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 gru/while/gru_cell/dropout_1/Mul
"gru/while/gru_cell/dropout_1/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_1/Shape
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2³¹m2;
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2-
+gru/while/gru_cell/dropout_1/GreaterEqual/y
)gru/while/gru_cell/dropout_1/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_1/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2+
)gru/while/gru_cell/dropout_1/GreaterEqual¿
!gru/while/gru_cell/dropout_1/CastCast-gru/while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru/while/gru_cell/dropout_1/CastÏ
"gru/while/gru_cell/dropout_1/Mul_1Mul$gru/while/gru_cell/dropout_1/Mul:z:0%gru/while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"gru/while/gru_cell/dropout_1/Mul_1
"gru/while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_2/ConstÒ
 gru/while/gru_cell/dropout_2/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 gru/while/gru_cell/dropout_2/Mul
"gru/while/gru_cell/dropout_2/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_2/Shape
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2´¡2;
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2-
+gru/while/gru_cell/dropout_2/GreaterEqual/y
)gru/while/gru_cell/dropout_2/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_2/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2+
)gru/while/gru_cell/dropout_2/GreaterEqual¿
!gru/while/gru_cell/dropout_2/CastCast-gru/while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru/while/gru_cell/dropout_2/CastÏ
"gru/while/gru_cell/dropout_2/Mul_1Mul$gru/while/gru_cell/dropout_2/Mul:z:0%gru/while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"gru/while/gru_cell/dropout_2/Mul_1
$gru/while/gru_cell/ones_like_1/ShapeShapegru_while_placeholder_3*
T0*
_output_shapes
:2&
$gru/while/gru_cell/ones_like_1/Shape
$gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$gru/while/gru_cell/ones_like_1/ConstØ
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell/ones_like_1
"gru/while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_3/ConstÓ
 gru/while/gru_cell/dropout_3/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gru/while/gru_cell/dropout_3/Mul
"gru/while/gru_cell/dropout_3/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_3/Shape
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Û°É2;
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2-
+gru/while/gru_cell/dropout_3/GreaterEqual/y
)gru/while/gru_cell/dropout_3/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_3/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)gru/while/gru_cell/dropout_3/GreaterEqual¾
!gru/while/gru_cell/dropout_3/CastCast-gru/while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru/while/gru_cell/dropout_3/CastÎ
"gru/while/gru_cell/dropout_3/Mul_1Mul$gru/while/gru_cell/dropout_3/Mul:z:0%gru/while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gru/while/gru_cell/dropout_3/Mul_1
"gru/while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_4/ConstÓ
 gru/while/gru_cell/dropout_4/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gru/while/gru_cell/dropout_4/Mul
"gru/while/gru_cell/dropout_4/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_4/Shape
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2è2;
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2-
+gru/while/gru_cell/dropout_4/GreaterEqual/y
)gru/while/gru_cell/dropout_4/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_4/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)gru/while/gru_cell/dropout_4/GreaterEqual¾
!gru/while/gru_cell/dropout_4/CastCast-gru/while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru/while/gru_cell/dropout_4/CastÎ
"gru/while/gru_cell/dropout_4/Mul_1Mul$gru/while/gru_cell/dropout_4/Mul:z:0%gru/while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gru/while/gru_cell/dropout_4/Mul_1
"gru/while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_5/ConstÓ
 gru/while/gru_cell/dropout_5/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gru/while/gru_cell/dropout_5/Mul
"gru/while/gru_cell/dropout_5/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_5/Shape
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ÌÂ2;
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2-
+gru/while/gru_cell/dropout_5/GreaterEqual/y
)gru/while/gru_cell/dropout_5/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_5/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)gru/while/gru_cell/dropout_5/GreaterEqual¾
!gru/while/gru_cell/dropout_5/CastCast-gru/while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gru/while/gru_cell/dropout_5/CastÎ
"gru/while/gru_cell/dropout_5/Mul_1Mul$gru/while/gru_cell/dropout_5/Mul:z:0%gru/while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gru/while/gru_cell/dropout_5/Mul_1³
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02#
!gru/while/gru_cell/ReadVariableOp£
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell/unstackÆ
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0$gru/while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/mulÌ
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/mul_1Ì
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/mul_2º
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1¡
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack¥
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice/stack_1¥
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2ñ
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice¹
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMulº
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2¥
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_1/stack©
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_1/stack_1©
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2û
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1Á
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_1º
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3¥
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(gru/while/gru_cell/strided_slice_2/stack©
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1©
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2û
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2Á
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_2
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack¢
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_3/stack_1¢
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2Þ
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_3Ç
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_4/stack¢
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_4/stack_1¢
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2Ì
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_4Í
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_1
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2*
(gru/while/gru_cell/strided_slice_5/stack¢
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1¢
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2Ü
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2$
"gru/while/gru_cell/strided_slice_5Í
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_2®
gru/while/gru_cell/mul_3Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_3®
gru/while/gru_cell/mul_4Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_4®
gru/while/gru_cell/mul_5Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_5¹
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4¥
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack©
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_6/stack_1©
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2ú
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6Á
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_3¹
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5¥
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_7/stack©
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_7/stack_1©
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2ú
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7Á
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_4
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stack¢
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_8/stack_1¢
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2Þ
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_8Í
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_3
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_9/stack¢
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_9/stack_1¢
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2Ì
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_9Í
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_4·
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid½
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_1
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid_1¹
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6§
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)gru/while/gru_cell/strided_slice_10/stack«
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1«
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2ÿ
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10Â
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_5 
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2+
)gru/while/gru_cell/strided_slice_11/stack¤
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1¤
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2á
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2%
#gru/while/gru_cell/strided_slice_11Î
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_5¶
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_6´
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_2
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Tanh¦
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell/sub/x¬
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/sub¦
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_8«
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_3
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
gru/while/Tile¶
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_2*
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
gru/while/Tile_1¼
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_3*
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
gru/while/add_1ò
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1ô
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2¡
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/SelectV2:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_4
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_5"^
,gru_while_gru_cell_readvariableop_1_resource.gru_while_gru_cell_readvariableop_1_resource_0"^
,gru_while_gru_cell_readvariableop_4_resource.gru_while_gru_cell_readvariableop_4_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
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
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_6: 
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
'
µ
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14048

inputs
	gru_13992:`
	gru_13994:	¬`
	gru_13996: `
output_14030: 
output_14032:
identity¢gru/StatefulPartitionedCall¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallã
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
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_13992	gru_13994	gru_13996*
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
>__inference_gru_layer_call_and_return_conditional_losses_139912
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14030output_14032*
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
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCall¹
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_13994*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÌ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_13996*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÁ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
"

while_body_13011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_gru_cell_13033_0:`)
while_gru_cell_13035_0:	¬`(
while_gru_cell_13037_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_13033:`'
while_gru_cell_13035:	¬`&
while_gru_cell_13037: `¢&while/gru_cell/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem 
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_13033_0while_gru_cell_13035_0while_gru_cell_13037_0*
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
GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_129982(
&while/gru_cell/StatefulPartitionedCalló
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¶
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¼
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4".
while_gru_cell_13033while_gru_cell_13033_0".
while_gru_cell_13035while_gru_cell_13035_0".
while_gru_cell_13037while_gru_cell_13037_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 
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

´
#__inference_gru_layer_call_fn_15537
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
>__inference_gru_layer_call_and_return_conditional_losses_134192
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
'
´
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14593	
input
	gru_14568:`
	gru_14570:	¬`
	gru_14572: `
output_14575: 
output_14577:
identity¢gru/StatefulPartitionedCall¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallâ
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
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_14568	gru_14570	gru_14572*
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
>__inference_gru_layer_call_and_return_conditional_losses_139912
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14575output_14577*
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
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCall¹
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14570*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÌ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14572*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÁ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
õ
C
'__inference_masking_layer_call_fn_15492

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
B__inference_masking_layer_call_and_return_conditional_losses_136952
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
ª²
Ì
while_body_13827
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	¬`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	¬`:
(while_gru_cell_readvariableop_4_resource: `¢while/gru_cell/ReadVariableOp¢while/gru_cell/ReadVariableOp_1¢while/gru_cell/ReadVariableOp_2¢while/gru_cell/ReadVariableOp_3¢while/gru_cell/ReadVariableOp_4¢while/gru_cell/ReadVariableOp_5¢while/gru_cell/ReadVariableOp_6Ã
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
)while/TensorArrayV2Read/TensorListGetItem 
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
while/gru_cell/ones_like/ConstÁ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/ones_like
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell/ones_like_1/ConstÈ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/ones_like_1§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack·
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul»
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_1»
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_2®
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_1
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2Ù
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul®
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack¡
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1¡
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2ã
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_1®
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack¡
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1¡
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2ã
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_2
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2Æ
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3·
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stack
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2´
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4½
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_1
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stack
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2Ä
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5½
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_5­
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack¡
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1¡
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2â
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6±
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_3­
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack¡
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1¡
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2â
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7±
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_4
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2Æ
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8½
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_3
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stack
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2´
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9½
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_4§
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1­
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/gru_cell/strided_slice_10/stack£
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1£
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ç
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10²
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_5
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stack
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2É
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11¾
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_5¦
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_6¤
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/add_1Ê
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityÝ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ì
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2ù
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3è
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
â
ò
.__inference_GRU_classifier_layer_call_fn_14564	
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_145362
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
Ü
¬
#GRU_classifier_gru_while_body_12656B
>gru_classifier_gru_while_gru_classifier_gru_while_loop_counterH
Dgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations(
$gru_classifier_gru_while_placeholder*
&gru_classifier_gru_while_placeholder_1*
&gru_classifier_gru_while_placeholder_2*
&gru_classifier_gru_while_placeholder_3A
=gru_classifier_gru_while_gru_classifier_gru_strided_slice_1_0}
ygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0
}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0M
;gru_classifier_gru_while_gru_cell_readvariableop_resource_0:`P
=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0:	¬`O
=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0: `%
!gru_classifier_gru_while_identity'
#gru_classifier_gru_while_identity_1'
#gru_classifier_gru_while_identity_2'
#gru_classifier_gru_while_identity_3'
#gru_classifier_gru_while_identity_4'
#gru_classifier_gru_while_identity_5?
;gru_classifier_gru_while_gru_classifier_gru_strided_slice_1{
wgru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor
{gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensorK
9gru_classifier_gru_while_gru_cell_readvariableop_resource:`N
;gru_classifier_gru_while_gru_cell_readvariableop_1_resource:	¬`M
;gru_classifier_gru_while_gru_cell_readvariableop_4_resource: `¢0GRU_classifier/gru/while/gru_cell/ReadVariableOp¢2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1¢2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2¢2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3¢2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4¢2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5¢2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6é
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
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemÙ
1GRU_classifier/gru/while/gru_cell/ones_like/ShapeShapeCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1GRU_classifier/gru/while/gru_cell/ones_like/Shape«
1GRU_classifier/gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1GRU_classifier/gru/while/gru_cell/ones_like/Const
+GRU_classifier/gru/while/gru_cell/ones_likeFill:GRU_classifier/gru/while/gru_cell/ones_like/Shape:output:0:GRU_classifier/gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2-
+GRU_classifier/gru/while/gru_cell/ones_likeÀ
3GRU_classifier/gru/while/gru_cell/ones_like_1/ShapeShape&gru_classifier_gru_while_placeholder_3*
T0*
_output_shapes
:25
3GRU_classifier/gru/while/gru_cell/ones_like_1/Shape¯
3GRU_classifier/gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3GRU_classifier/gru/while/gru_cell/ones_like_1/Const
-GRU_classifier/gru/while/gru_cell/ones_like_1Fill<GRU_classifier/gru/while/gru_cell/ones_like_1/Shape:output:0<GRU_classifier/gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-GRU_classifier/gru/while/gru_cell/ones_like_1à
0GRU_classifier/gru/while/gru_cell/ReadVariableOpReadVariableOp;gru_classifier_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype022
0GRU_classifier/gru/while/gru_cell/ReadVariableOpÐ
)GRU_classifier/gru/while/gru_cell/unstackUnpack8GRU_classifier/gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2+
)GRU_classifier/gru/while/gru_cell/unstack
%GRU_classifier/gru/while/gru_cell/mulMulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%GRU_classifier/gru/while/gru_cell/mul
'GRU_classifier/gru/while/gru_cell/mul_1MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'GRU_classifier/gru/while/gru_cell/mul_1
'GRU_classifier/gru/while/gru_cell/mul_2MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'GRU_classifier/gru/while/gru_cell/mul_2ç
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1¿
5GRU_classifier/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5GRU_classifier/gru/while/gru_cell/strided_slice/stackÃ
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_1Ã
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_2Ë
/GRU_classifier/gru/while/gru_cell/strided_sliceStridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_1:value:0>GRU_classifier/gru/while/gru_cell/strided_slice/stack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice/stack_1:output:0@GRU_classifier/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask21
/GRU_classifier/gru/while/gru_cell/strided_sliceõ
(GRU_classifier/gru/while/gru_cell/MatMulMatMul)GRU_classifier/gru/while/gru_cell/mul:z:08GRU_classifier/gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(GRU_classifier/gru/while/gru_cell/MatMulç
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2Ã
7GRU_classifier/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_1/stackÇ
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1Ç
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2Õ
1GRU_classifier/gru/while/gru_cell/strided_slice_1StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_2:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_1/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_1ý
*GRU_classifier/gru/while/gru_cell/MatMul_1MatMul+GRU_classifier/gru/while/gru_cell/mul_1:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_1ç
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3Ã
7GRU_classifier/gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7GRU_classifier/gru/while/gru_cell/strided_slice_2/stackÇ
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1Ç
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2Õ
1GRU_classifier/gru/while/gru_cell/strided_slice_2StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_3:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_2/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_2ý
*GRU_classifier/gru/while/gru_cell/MatMul_2MatMul+GRU_classifier/gru/while/gru_cell/mul_2:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_2¼
7GRU_classifier/gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_3/stackÀ
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1À
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2¸
1GRU_classifier/gru/while/gru_cell/strided_slice_3StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_3/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_3
)GRU_classifier/gru/while/gru_cell/BiasAddBiasAdd2GRU_classifier/gru/while/gru_cell/MatMul:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell/BiasAdd¼
7GRU_classifier/gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_4/stackÀ
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1À
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2¦
1GRU_classifier/gru/while/gru_cell/strided_slice_4StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_4/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 23
1GRU_classifier/gru/while/gru_cell/strided_slice_4
+GRU_classifier/gru/while/gru_cell/BiasAdd_1BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_1:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_1¼
7GRU_classifier/gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@29
7GRU_classifier/gru/while/gru_cell/strided_slice_5/stackÀ
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1À
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2¶
1GRU_classifier/gru/while/gru_cell/strided_slice_5StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_5/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_5
+GRU_classifier/gru/while/gru_cell/BiasAdd_2BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_2:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_2ë
'GRU_classifier/gru/while/gru_cell/mul_3Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/mul_3ë
'GRU_classifier/gru/while/gru_cell/mul_4Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/mul_4ë
'GRU_classifier/gru/while/gru_cell/mul_5Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/mul_5æ
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4Ã
7GRU_classifier/gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_6/stackÇ
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1Ç
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2Ô
1GRU_classifier/gru/while/gru_cell/strided_slice_6StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_4:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_6/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_6ý
*GRU_classifier/gru/while/gru_cell/MatMul_3MatMul+GRU_classifier/gru/while/gru_cell/mul_3:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_3æ
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5Ã
7GRU_classifier/gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_7/stackÇ
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1Ç
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2Ô
1GRU_classifier/gru/while/gru_cell/strided_slice_7StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_5:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_7/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_7ý
*GRU_classifier/gru/while/gru_cell/MatMul_4MatMul+GRU_classifier/gru/while/gru_cell/mul_4:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_4¼
7GRU_classifier/gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_8/stackÀ
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1À
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2¸
1GRU_classifier/gru/while/gru_cell/strided_slice_8StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1@GRU_classifier/gru/while/gru_cell/strided_slice_8/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_8
+GRU_classifier/gru/while/gru_cell/BiasAdd_3BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_3:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_3¼
7GRU_classifier/gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_9/stackÀ
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1À
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2¦
1GRU_classifier/gru/while/gru_cell/strided_slice_9StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1@GRU_classifier/gru/while/gru_cell/strided_slice_9/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 23
1GRU_classifier/gru/while/gru_cell/strided_slice_9
+GRU_classifier/gru/while/gru_cell/BiasAdd_4BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_4:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_4ó
%GRU_classifier/gru/while/gru_cell/addAddV22GRU_classifier/gru/while/gru_cell/BiasAdd:output:04GRU_classifier/gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/while/gru_cell/add¾
)GRU_classifier/gru/while/gru_cell/SigmoidSigmoid)GRU_classifier/gru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)GRU_classifier/gru/while/gru_cell/Sigmoidù
'GRU_classifier/gru/while/gru_cell/add_1AddV24GRU_classifier/gru/while/gru_cell/BiasAdd_1:output:04GRU_classifier/gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/add_1Ä
+GRU_classifier/gru/while/gru_cell/Sigmoid_1Sigmoid+GRU_classifier/gru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell/Sigmoid_1æ
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6Å
8GRU_classifier/gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2:
8GRU_classifier/gru/while/gru_cell/strided_slice_10/stackÉ
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1É
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2Ù
2GRU_classifier/gru/while/gru_cell/strided_slice_10StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_6:value:0AGRU_classifier/gru/while/gru_cell/strided_slice_10/stack:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2GRU_classifier/gru/while/gru_cell/strided_slice_10þ
*GRU_classifier/gru/while/gru_cell/MatMul_5MatMul+GRU_classifier/gru/while/gru_cell/mul_5:z:0;GRU_classifier/gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_5¾
8GRU_classifier/gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2:
8GRU_classifier/gru/while/gru_cell/strided_slice_11/stackÂ
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1Â
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2»
2GRU_classifier/gru/while/gru_cell/strided_slice_11StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1AGRU_classifier/gru/while/gru_cell/strided_slice_11/stack:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask24
2GRU_classifier/gru/while/gru_cell/strided_slice_11
+GRU_classifier/gru/while/gru_cell/BiasAdd_5BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_5:product:0;GRU_classifier/gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_5ò
'GRU_classifier/gru/while/gru_cell/mul_6Mul/GRU_classifier/gru/while/gru_cell/Sigmoid_1:y:04GRU_classifier/gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/mul_6ð
'GRU_classifier/gru/while/gru_cell/add_2AddV24GRU_classifier/gru/while/gru_cell/BiasAdd_2:output:0+GRU_classifier/gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/add_2·
&GRU_classifier/gru/while/gru_cell/TanhTanh+GRU_classifier/gru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&GRU_classifier/gru/while/gru_cell/Tanhâ
'GRU_classifier/gru/while/gru_cell/mul_7Mul-GRU_classifier/gru/while/gru_cell/Sigmoid:y:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/mul_7
'GRU_classifier/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'GRU_classifier/gru/while/gru_cell/sub/xè
%GRU_classifier/gru/while/gru_cell/subSub0GRU_classifier/gru/while/gru_cell/sub/x:output:0-GRU_classifier/gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/while/gru_cell/subâ
'GRU_classifier/gru/while/gru_cell/mul_8Mul)GRU_classifier/gru/while/gru_cell/sub:z:0*GRU_classifier/gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/mul_8ç
'GRU_classifier/gru/while/gru_cell/add_3AddV2+GRU_classifier/gru/while/gru_cell/mul_7:z:0+GRU_classifier/gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/while/gru_cell/add_3£
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
GRU_classifier/gru/while/Tile
!GRU_classifier/gru/while/SelectV2SelectV2&GRU_classifier/gru/while/Tile:output:0+GRU_classifier/gru/while/gru_cell/add_3:z:0&gru_classifier_gru_while_placeholder_2*
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
GRU_classifier/gru/while/Tile_1
#GRU_classifier/gru/while/SelectV2_1SelectV2(GRU_classifier/gru/while/Tile_1:output:0+GRU_classifier/gru/while/gru_cell/add_3:z:0&gru_classifier_gru_while_placeholder_3*
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
GRU_classifier/gru/while/add_1
!GRU_classifier/gru/while/IdentityIdentity"GRU_classifier/gru/while/add_1:z:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2#
!GRU_classifier/gru/while/Identity®
#GRU_classifier/gru/while/Identity_1IdentityDgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations1^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_1
#GRU_classifier/gru/while/Identity_2Identity GRU_classifier/gru/while/add:z:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_2·
#GRU_classifier/gru/while/Identity_3IdentityMGRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_3¥
#GRU_classifier/gru/while/Identity_4Identity*GRU_classifier/gru/while/SelectV2:output:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/while/Identity_4§
#GRU_classifier/gru/while/Identity_5Identity,GRU_classifier/gru/while/SelectV2_1:output:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/while/Identity_5"|
;gru_classifier_gru_while_gru_cell_readvariableop_1_resource=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0"|
;gru_classifier_gru_while_gru_cell_readvariableop_4_resource=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0"x
9gru_classifier_gru_while_gru_cell_readvariableop_resource;gru_classifier_gru_while_gru_cell_readvariableop_resource_0"|
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
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 2d
0GRU_classifier/gru/while/gru_cell/ReadVariableOp0GRU_classifier/gru/while/gru_cell/ReadVariableOp2h
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_12GRU_classifier/gru/while/gru_cell/ReadVariableOp_12h
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_22GRU_classifier/gru/while/gru_cell/ReadVariableOp_22h
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_32GRU_classifier/gru/while/gru_cell/ReadVariableOp_32h
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_42GRU_classifier/gru/while/gru_cell/ReadVariableOp_42h
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_52GRU_classifier/gru/while/gru_cell/ReadVariableOp_52h
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_62GRU_classifier/gru/while/gru_cell/ReadVariableOp_6: 
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
ÄÙ
­
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15487

inputs6
$gru_gru_cell_readvariableop_resource:`9
&gru_gru_cell_readvariableop_1_resource:	¬`8
&gru_gru_cell_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identity¢gru/gru_cell/ReadVariableOp¢gru/gru_cell/ReadVariableOp_1¢gru/gru_cell/ReadVariableOp_2¢gru/gru_cell/ReadVariableOp_3¢gru/gru_cell/ReadVariableOp_4¢gru/gru_cell/ReadVariableOp_5¢gru/gru_cell/ReadVariableOp_6¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢	gru/while¢output/BiasAdd/ReadVariableOp¢output/Tensordot/ReadVariableOpm
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
gru/strided_slice_2
gru/gru_cell/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/Shape
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell/ones_like/Const¹
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/ones_like}
gru/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout/Const´
gru/gru_cell/dropout/MulMulgru/gru_cell/ones_like:output:0#gru/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout/Mul
gru/gru_cell/dropout/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout/Shapeø
1gru/gru_cell/dropout/random_uniform/RandomUniformRandomUniform#gru/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÂÎ23
1gru/gru_cell/dropout/random_uniform/RandomUniform
#gru/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2%
#gru/gru_cell/dropout/GreaterEqual/yó
!gru/gru_cell/dropout/GreaterEqualGreaterEqual:gru/gru_cell/dropout/random_uniform/RandomUniform:output:0,gru/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!gru/gru_cell/dropout/GreaterEqual§
gru/gru_cell/dropout/CastCast%gru/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout/Cast¯
gru/gru_cell/dropout/Mul_1Mulgru/gru_cell/dropout/Mul:z:0gru/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout/Mul_1
gru/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_1/Constº
gru/gru_cell/dropout_1/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout_1/Mul
gru/gru_cell/dropout_1/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_1/Shapeþ
3gru/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2×Ç25
3gru/gru_cell/dropout_1/random_uniform/RandomUniform
%gru/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%gru/gru_cell/dropout_1/GreaterEqual/yû
#gru/gru_cell/dropout_1/GreaterEqualGreaterEqual<gru/gru_cell/dropout_1/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#gru/gru_cell/dropout_1/GreaterEqual­
gru/gru_cell/dropout_1/CastCast'gru/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout_1/Cast·
gru/gru_cell/dropout_1/Mul_1Mulgru/gru_cell/dropout_1/Mul:z:0gru/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout_1/Mul_1
gru/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_2/Constº
gru/gru_cell/dropout_2/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout_2/Mul
gru/gru_cell/dropout_2/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_2/Shapeþ
3gru/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2¿àõ25
3gru/gru_cell/dropout_2/random_uniform/RandomUniform
%gru/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%gru/gru_cell/dropout_2/GreaterEqual/yû
#gru/gru_cell/dropout_2/GreaterEqualGreaterEqual<gru/gru_cell/dropout_2/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#gru/gru_cell/dropout_2/GreaterEqual­
gru/gru_cell/dropout_2/CastCast'gru/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout_2/Cast·
gru/gru_cell/dropout_2/Mul_1Mulgru/gru_cell/dropout_2/Mul:z:0gru/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/dropout_2/Mul_1
gru/gru_cell/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2 
gru/gru_cell/ones_like_1/Shape
gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
gru/gru_cell/ones_like_1/ConstÀ
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/ones_like_1
gru/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_3/Const»
gru/gru_cell/dropout_3/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_3/Mul
gru/gru_cell/dropout_3/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_3/Shapeý
3gru/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ý25
3gru/gru_cell/dropout_3/random_uniform/RandomUniform
%gru/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%gru/gru_cell/dropout_3/GreaterEqual/yú
#gru/gru_cell/dropout_3/GreaterEqualGreaterEqual<gru/gru_cell/dropout_3/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gru/gru_cell/dropout_3/GreaterEqual¬
gru/gru_cell/dropout_3/CastCast'gru/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_3/Cast¶
gru/gru_cell/dropout_3/Mul_1Mulgru/gru_cell/dropout_3/Mul:z:0gru/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_3/Mul_1
gru/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_4/Const»
gru/gru_cell/dropout_4/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_4/Mul
gru/gru_cell/dropout_4/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_4/Shapeü
3gru/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2³¼25
3gru/gru_cell/dropout_4/random_uniform/RandomUniform
%gru/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%gru/gru_cell/dropout_4/GreaterEqual/yú
#gru/gru_cell/dropout_4/GreaterEqualGreaterEqual<gru/gru_cell/dropout_4/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gru/gru_cell/dropout_4/GreaterEqual¬
gru/gru_cell/dropout_4/CastCast'gru/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_4/Cast¶
gru/gru_cell/dropout_4/Mul_1Mulgru/gru_cell/dropout_4/Mul:z:0gru/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_4/Mul_1
gru/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_5/Const»
gru/gru_cell/dropout_5/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_5/Mul
gru/gru_cell/dropout_5/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_5/Shapeü
3gru/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2â(25
3gru/gru_cell/dropout_5/random_uniform/RandomUniform
%gru/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%gru/gru_cell/dropout_5/GreaterEqual/yú
#gru/gru_cell/dropout_5/GreaterEqualGreaterEqual<gru/gru_cell/dropout_5/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gru/gru_cell/dropout_5/GreaterEqual¬
gru/gru_cell/dropout_5/CastCast'gru/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_5/Cast¶
gru/gru_cell/dropout_5/Mul_1Mulgru/gru_cell/dropout_5/Mul:z:0gru/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/dropout_5/Mul_1
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell/ReadVariableOp
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell/unstack
gru/gru_cell/mulMulgru/strided_slice_2:output:0gru/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/mul¢
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/mul_1¢
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/mul_2¦
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru/gru_cell/ReadVariableOp_1
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice/stack_1
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2Í
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice¡
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul¦
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru/gru_cell/ReadVariableOp_2
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_1/stack
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_1/stack_1
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2×
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1©
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_1¦
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru/gru_cell/ReadVariableOp_3
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru/gru_cell/strided_slice_2/stack
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2×
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2©
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_2
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_3/stack_1
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2º
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_3¯
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_4/stack
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_4/stack_1
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2¨
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_4µ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_1
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2$
"gru/gru_cell/strided_slice_5/stack
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2¸
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_5µ
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_2
gru/gru_cell/mul_3Mulgru/zeros:output:0 gru/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_3
gru/gru_cell/mul_4Mulgru/zeros:output:0 gru/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_4
gru/gru_cell/mul_5Mulgru/zeros:output:0 gru/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_5¥
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_4
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_6/stack_1
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2Ö
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6©
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul_3:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_3¥
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_5
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_7/stack
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_7/stack_1
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2Ö
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7©
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_4:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_4
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stack
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_8/stack_1
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2º
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_8µ
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_3
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_9/stack
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_9/stack_1
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2¨
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_9µ
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_4
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid¥
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_1
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid_1¥
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_6
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#gru/gru_cell/strided_slice_10/stack
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2Û
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10ª
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_5
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2%
#gru/gru_cell/strided_slice_11/stack
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2½
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_11¶
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_5
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_6
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Tanh
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_7m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell/sub/x
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/sub
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_8
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_3
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
-gru/TensorArrayUnstack_1/TensorListFromTensorw
gru/zeros_like	ZerosLikegru/gru_cell/add_3:z:0*
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
gru/while/loop_counterÊ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
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
gru_while_body_15234* 
condR
gru_while_cond_15233*M
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
output/BiasAddÖ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulé
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2
	gru/while	gru/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2B
output/Tensordot/ReadVariableOpoutput/Tensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
÷ù
Ì
while_body_16033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	¬`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	¬`:
(while_gru_cell_readvariableop_4_resource: `¢while/gru_cell/ReadVariableOp¢while/gru_cell/ReadVariableOp_1¢while/gru_cell/ReadVariableOp_2¢while/gru_cell/ReadVariableOp_3¢while/gru_cell/ReadVariableOp_4¢while/gru_cell/ReadVariableOp_5¢while/gru_cell/ReadVariableOp_6Ã
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
)while/TensorArrayV2Read/TensorListGetItem 
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
while/gru_cell/ones_like/ConstÁ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/ones_like
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/Const¼
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Mul
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeþ
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2é¥§25
3while/gru_cell/dropout/random_uniform/RandomUniform
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%while/gru_cell/dropout/GreaterEqual/yû
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#while/gru_cell/dropout/GreaterEqual­
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Cast·
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Mul_1
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/ConstÂ
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_1/Mul
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2éËÛ27
5while/gru_cell/dropout_1/random_uniform/RandomUniform
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_1/GreaterEqual/y
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell/dropout_1/GreaterEqual³
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_1/Cast¿
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell/dropout_1/Mul_1
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/ConstÂ
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_2/Mul
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2Ô»Þ27
5while/gru_cell/dropout_2/random_uniform/RandomUniform
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_2/GreaterEqual/y
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell/dropout_2/GreaterEqual³
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_2/Cast¿
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell/dropout_2/Mul_1
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell/ones_like_1/ConstÈ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/ones_like_1
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_3/ConstÃ
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_3/Mul
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ïÛÎ27
5while/gru_cell/dropout_3/random_uniform/RandomUniform
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_3/GreaterEqual/y
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_3/GreaterEqual²
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_3/Cast¾
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_3/Mul_1
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_4/ConstÃ
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_4/Mul
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2úàÙ27
5while/gru_cell/dropout_4/random_uniform/RandomUniform
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_4/GreaterEqual/y
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_4/GreaterEqual²
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_4/Cast¾
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_4/Mul_1
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_5/ConstÃ
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_5/Mul
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ôëþ27
5while/gru_cell/dropout_5/random_uniform/RandomUniform
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_5/GreaterEqual/y
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_5/GreaterEqual²
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_5/Cast¾
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_5/Mul_1§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¶
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul¼
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_1¼
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_2®
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_1
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2Ù
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul®
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack¡
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1¡
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2ã
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_1®
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack¡
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1¡
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2ã
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_2
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2Æ
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3·
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stack
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2´
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4½
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_1
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stack
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2Ä
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5½
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_5­
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack¡
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1¡
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2â
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6±
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_3­
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack¡
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1¡
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2â
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7±
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_4
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2Æ
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8½
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_3
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stack
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2´
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9½
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_4§
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1­
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/gru_cell/strided_slice_10/stack£
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1£
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ç
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10²
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_5
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stack
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2É
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11¾
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_5¦
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_6¤
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/add_1Ê
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityÝ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ì
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2ù
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3è
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
&__inference_output_layer_call_fn_16940

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
A__inference_output_layer_call_and_return_conditional_losses_140292
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
õ
¥
while_cond_13010
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_13010___redundant_placeholder03
/while_while_cond_13010___redundant_placeholder13
/while_while_cond_13010___redundant_placeholder23
/while_while_cond_13010___redundant_placeholder3
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
õù
Ì
while_body_14263
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	¬`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	¬`:
(while_gru_cell_readvariableop_4_resource: `¢while/gru_cell/ReadVariableOp¢while/gru_cell/ReadVariableOp_1¢while/gru_cell/ReadVariableOp_2¢while/gru_cell/ReadVariableOp_3¢while/gru_cell/ReadVariableOp_4¢while/gru_cell/ReadVariableOp_5¢while/gru_cell/ReadVariableOp_6Ã
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
)while/TensorArrayV2Read/TensorListGetItem 
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
while/gru_cell/ones_like/ConstÁ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/ones_like
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/Const¼
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Mul
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeþ
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÀÐØ25
3while/gru_cell/dropout/random_uniform/RandomUniform
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%while/gru_cell/dropout/GreaterEqual/yû
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#while/gru_cell/dropout/GreaterEqual­
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Cast·
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Mul_1
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/ConstÂ
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_1/Mul
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ÌÀ 27
5while/gru_cell/dropout_1/random_uniform/RandomUniform
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_1/GreaterEqual/y
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell/dropout_1/GreaterEqual³
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_1/Cast¿
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell/dropout_1/Mul_1
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/ConstÂ
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_2/Mul
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ûÞ×27
5while/gru_cell/dropout_2/random_uniform/RandomUniform
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_2/GreaterEqual/y
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell/dropout_2/GreaterEqual³
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_2/Cast¿
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell/dropout_2/Mul_1
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell/ones_like_1/ConstÈ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/ones_like_1
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_3/ConstÃ
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_3/Mul
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Îµ®27
5while/gru_cell/dropout_3/random_uniform/RandomUniform
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_3/GreaterEqual/y
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_3/GreaterEqual²
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_3/Cast¾
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_3/Mul_1
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_4/ConstÃ
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_4/Mul
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed227
5while/gru_cell/dropout_4/random_uniform/RandomUniform
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_4/GreaterEqual/y
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_4/GreaterEqual²
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_4/Cast¾
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_4/Mul_1
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_5/ConstÃ
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_5/Mul
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ä§27
5while/gru_cell/dropout_5/random_uniform/RandomUniform
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_5/GreaterEqual/y
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_5/GreaterEqual²
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_5/Cast¾
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_5/Mul_1§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¶
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul¼
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_1¼
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_2®
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_1
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2Ù
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul®
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack¡
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1¡
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2ã
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_1®
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack¡
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1¡
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2ã
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_2
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2Æ
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3·
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stack
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2´
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4½
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_1
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stack
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2Ä
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5½
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_5­
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack¡
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1¡
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2â
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6±
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_3­
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack¡
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1¡
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2â
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7±
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_4
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2Æ
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8½
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_3
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stack
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2´
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9½
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_4§
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1­
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/gru_cell/strided_slice_10/stack£
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1£
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ç
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10²
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_5
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stack
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2É
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11¾
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_5¦
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_6¤
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/add_1Ê
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityÝ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ì
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2ù
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3è
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
ª

Ë
gru_while_cond_14833$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_14833___redundant_placeholder0;
7gru_while_gru_while_cond_14833___redundant_placeholder1;
7gru_while_gru_while_cond_14833___redundant_placeholder2;
7gru_while_gru_while_cond_14833___redundant_placeholder3;
7gru_while_gru_while_cond_14833___redundant_placeholder4
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
Â 
ø
A__inference_output_layer_call_and_return_conditional_losses_14029

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
ª²
Ì
while_body_16376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	¬`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	¬`:
(while_gru_cell_readvariableop_4_resource: `¢while/gru_cell/ReadVariableOp¢while/gru_cell/ReadVariableOp_1¢while/gru_cell/ReadVariableOp_2¢while/gru_cell/ReadVariableOp_3¢while/gru_cell/ReadVariableOp_4¢while/gru_cell/ReadVariableOp_5¢while/gru_cell/ReadVariableOp_6Ã
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
)while/TensorArrayV2Read/TensorListGetItem 
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
while/gru_cell/ones_like/ConstÁ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/ones_like
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell/ones_like_1/ConstÈ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/ones_like_1§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack·
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul»
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_1»
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_2®
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_1
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2Ù
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul®
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack¡
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1¡
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2ã
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_1®
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack¡
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1¡
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2ã
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_2
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2Æ
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3·
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stack
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2´
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4½
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_1
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stack
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2Ä
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5½
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_5­
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack¡
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1¡
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2â
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6±
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_3­
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack¡
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1¡
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2â
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7±
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_4
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2Æ
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8½
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_3
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stack
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2´
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9½
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_4§
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1­
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/gru_cell/strided_slice_10/stack£
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1£
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ç
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10²
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_5
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stack
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2É
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11¾
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_5¦
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_6¤
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/add_1Ê
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityÝ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ì
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2ù
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3è
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
â
Ä
>__inference_gru_layer_call_and_return_conditional_losses_16245
inputs_02
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	¬`4
"gru_cell_readvariableop_4_resource: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell/ReadVariableOp¢gru_cell/ReadVariableOp_1¢gru_cell/ReadVariableOp_2¢gru_cell/ReadVariableOp_3¢gru_cell/ReadVariableOp_4¢gru_cell/ReadVariableOp_5¢gru_cell/ReadVariableOp_6¢whileF
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
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/Const¤
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeì
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2×½Ì2/
-gru_cell/dropout/random_uniform/RandomUniform
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2!
gru_cell/dropout/GreaterEqual/yã
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/GreaterEqual
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Cast
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/Constª
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeò
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2óô21
/gru_cell/dropout_1/random_uniform/RandomUniform
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_1/GreaterEqual/yë
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell/dropout_1/GreaterEqual¡
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Cast§
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/Constª
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeò
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2¿ä21
/gru_cell/dropout_2/random_uniform/RandomUniform
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_2/GreaterEqual/yë
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell/dropout_2/GreaterEqual¡
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Cast§
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Mul_1v
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like_1/Const°
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_3/Const«
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Mul
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeñ
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2¯ÞÕ21
/gru_cell/dropout_3/random_uniform/RandomUniform
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_3/GreaterEqual/yê
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_3/GreaterEqual 
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Cast¦
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_4/Const«
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Mul
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeñ
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ã21
/gru_cell/dropout_4/random_uniform/RandomUniform
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_4/GreaterEqual/yê
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_4/GreaterEqual 
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Cast¦
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_5/Const«
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Mul
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeñ
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed221
/gru_cell/dropout_5/random_uniform/RandomUniform
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_5/GreaterEqual/yê
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_5/GreaterEqual 
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Cast¦
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Mul_1
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_1
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2µ
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_2
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stack
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2¿
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_3
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stack
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2¿
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_2
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2¢
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stack
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4¥
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_1
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stack
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2 
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5¥
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_5
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2¾
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_3
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stack
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2¾
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_4
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2¢
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8¥
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_3
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stack
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9¥
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stack
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2Ã
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_5
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stack
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2¥
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¦
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_16033*
condR
while_cond_16032*8
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
runtimeÒ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulå
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul´
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
Ú
Â
>__inference_gru_layer_call_and_return_conditional_losses_14475

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	¬`4
"gru_cell_readvariableop_4_resource: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell/ReadVariableOp¢gru_cell/ReadVariableOp_1¢gru_cell/ReadVariableOp_2¢gru_cell/ReadVariableOp_3¢gru_cell/ReadVariableOp_4¢gru_cell/ReadVariableOp_5¢gru_cell/ReadVariableOp_6¢whileD
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
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/Const¤
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeì
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2æµ2/
-gru_cell/dropout/random_uniform/RandomUniform
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2!
gru_cell/dropout/GreaterEqual/yã
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/GreaterEqual
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Cast
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/Constª
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeò
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ôßù21
/gru_cell/dropout_1/random_uniform/RandomUniform
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_1/GreaterEqual/yë
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell/dropout_1/GreaterEqual¡
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Cast§
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/Constª
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeò
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2 è21
/gru_cell/dropout_2/random_uniform/RandomUniform
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_2/GreaterEqual/yë
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell/dropout_2/GreaterEqual¡
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Cast§
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Mul_1v
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like_1/Const°
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_3/Const«
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Mul
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeñ
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ÿ»21
/gru_cell/dropout_3/random_uniform/RandomUniform
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_3/GreaterEqual/yê
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_3/GreaterEqual 
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Cast¦
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_4/Const«
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Mul
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeñ
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ñô­21
/gru_cell/dropout_4/random_uniform/RandomUniform
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_4/GreaterEqual/yê
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_4/GreaterEqual 
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Cast¦
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_5/Const«
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Mul
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeñ
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ãèü21
/gru_cell/dropout_5/random_uniform/RandomUniform
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_5/GreaterEqual/yê
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_5/GreaterEqual 
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Cast¦
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Mul_1
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_1
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2µ
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_2
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stack
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2¿
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_3
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stack
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2¿
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_2
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2¢
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stack
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4¥
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_1
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stack
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2 
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5¥
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_5
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2¾
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_3
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stack
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2¾
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_4
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2¢
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8¥
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_3
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stack
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9¥
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stack
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2Ã
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_5
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stack
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2¥
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¦
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_14263*
condR
while_cond_14262*8
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
runtimeÒ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulå
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul´
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs


#GRU_classifier_gru_while_cond_12655B
>gru_classifier_gru_while_gru_classifier_gru_while_loop_counterH
Dgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations(
$gru_classifier_gru_while_placeholder*
&gru_classifier_gru_while_placeholder_1*
&gru_classifier_gru_while_placeholder_2*
&gru_classifier_gru_while_placeholder_3D
@gru_classifier_gru_while_less_gru_classifier_gru_strided_slice_1Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_12655___redundant_placeholder0Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_12655___redundant_placeholder1Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_12655___redundant_placeholder2Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_12655___redundant_placeholder3Y
Ugru_classifier_gru_while_gru_classifier_gru_while_cond_12655___redundant_placeholder4%
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
:
²»

C__inference_gru_cell_layer_call_and_return_conditional_losses_17286

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
seed2ú2&
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
seed2Û£×2(
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
seed2[2(
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
seed2¤×2(
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
seed22(
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
seed2à½2(
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
add_3É
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÜ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÚ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÞ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
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
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
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
û
²
#__inference_gru_layer_call_fn_15548

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
>__inference_gru_layer_call_and_return_conditional_losses_139912
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
õ
¥
while_cond_16032
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_16032___redundant_placeholder03
/while_while_cond_16032___redundant_placeholder13
/while_while_cond_16032___redundant_placeholder23
/while_while_cond_16032___redundant_placeholder3
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
Â 
ø
A__inference_output_layer_call_and_return_conditional_losses_16970

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
û
²
#__inference_gru_layer_call_fn_15559

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
>__inference_gru_layer_call_and_return_conditional_losses_144752
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
õ
¥
while_cond_16375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_16375___redundant_placeholder03
/while_while_cond_16375___redundant_placeholder13
/while_while_cond_16375___redundant_placeholder23
/while_while_cond_16375___redundant_placeholder3
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
¬

Õ
(__inference_gru_cell_layer_call_fn_16996

inputs
states_0
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity

identity_1¢StatefulPartitionedCall¤
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
GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_129982
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
ú8
§

__inference__traced_save_17403
file_prefix,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_m_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_m_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_v_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_v_read_readvariableop
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
SaveV2/shape_and_slices®

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
¢Î
Ä
>__inference_gru_layer_call_and_return_conditional_losses_15854
inputs_02
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	¬`4
"gru_cell_readvariableop_4_resource: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell/ReadVariableOp¢gru_cell/ReadVariableOp_1¢gru_cell/ReadVariableOp_2¢gru_cell/ReadVariableOp_3¢gru_cell/ReadVariableOp_4¢gru_cell/ReadVariableOp_5¢gru_cell/ReadVariableOp_6¢whileF
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
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/ones_likev
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like_1/Const°
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/ones_like_1
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_1
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2µ
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_2
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stack
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2¿
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_3
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stack
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2¿
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_2
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2¢
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stack
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4¥
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_1
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stack
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2 
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5¥
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_5
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2¾
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_3
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stack
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2¾
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_4
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2¢
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8¥
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_3
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stack
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9¥
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stack
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2Ã
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_5
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stack
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2¥
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¦
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_15690*
condR
while_cond_15689*8
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
runtimeÒ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulå
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul´
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
Òh
Ó
!__inference__traced_restore_17485
file_prefix0
assignvariableop_output_kernel: ,
assignvariableop_1_output_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: 9
&assignvariableop_7_gru_gru_cell_kernel:	¬`B
0assignvariableop_8_gru_gru_cell_recurrent_kernel: `6
$assignvariableop_9_gru_gru_cell_bias:`#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: :
(assignvariableop_14_adam_output_kernel_m: 4
&assignvariableop_15_adam_output_bias_m:A
.assignvariableop_16_adam_gru_gru_cell_kernel_m:	¬`J
8assignvariableop_17_adam_gru_gru_cell_recurrent_kernel_m: `>
,assignvariableop_18_adam_gru_gru_cell_bias_m:`:
(assignvariableop_19_adam_output_kernel_v: 4
&assignvariableop_20_adam_output_bias_v:A
.assignvariableop_21_adam_gru_gru_cell_kernel_v:	¬`J
8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_v: `>
,assignvariableop_23_adam_gru_gru_cell_bias_v:`
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

Identity_7«
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gru_gru_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8µ
AssignVariableOp_8AssignVariableOp0assignvariableop_8_gru_gru_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_gru_gru_cell_biasIdentity_9:output:0"/device:CPU:0*
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
Identity_16¶
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_gru_gru_cell_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17À
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_gru_gru_cell_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18´
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_gru_gru_cell_bias_mIdentity_18:output:0"/device:CPU:0*
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
Identity_21¶
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_gru_gru_cell_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22À
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23´
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_gru_gru_cell_bias_vIdentity_23:output:0"/device:CPU:0*
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
²
Ì
__inference_loss_fn_1_17308Z
Hgru_gru_cell_recurrent_kernel_regularizer_square_readvariableop_resource: `
identity¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpHgru_gru_cell_recurrent_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul¶
IdentityIdentity1gru/gru_cell/recurrent_kernel/Regularizer/mul:z:0@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
¬

Õ
(__inference_gru_cell_layer_call_fn_17010

inputs
states_0
unknown:`
	unknown_0:	¬`
	unknown_1: `
identity

identity_1¢StatefulPartitionedCall¤
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
GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_132762
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
Ø
Â
>__inference_gru_layer_call_and_return_conditional_losses_16931

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	¬`4
"gru_cell_readvariableop_4_resource: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell/ReadVariableOp¢gru_cell/ReadVariableOp_1¢gru_cell/ReadVariableOp_2¢gru_cell/ReadVariableOp_3¢gru_cell/ReadVariableOp_4¢gru_cell/ReadVariableOp_5¢gru_cell/ReadVariableOp_6¢whileD
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
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/Const¤
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeë
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ó¡p2/
-gru_cell/dropout/random_uniform/RandomUniform
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2!
gru_cell/dropout/GreaterEqual/yã
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/GreaterEqual
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Cast
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/Constª
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeò
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2¶Èù21
/gru_cell/dropout_1/random_uniform/RandomUniform
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_1/GreaterEqual/yë
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell/dropout_1/GreaterEqual¡
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Cast§
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/Constª
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeò
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ñ±­21
/gru_cell/dropout_2/random_uniform/RandomUniform
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_2/GreaterEqual/yë
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
gru_cell/dropout_2/GreaterEqual¡
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Cast§
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/dropout_2/Mul_1v
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like_1/Const°
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_3/Const«
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Mul
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeñ
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2¹Ï21
/gru_cell/dropout_3/random_uniform/RandomUniform
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_3/GreaterEqual/yê
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_3/GreaterEqual 
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Cast¦
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_4/Const«
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Mul
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeñ
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ï§21
/gru_cell/dropout_4/random_uniform/RandomUniform
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_4/GreaterEqual/yê
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_4/GreaterEqual 
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Cast¦
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_5/Const«
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Mul
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeð
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2]21
/gru_cell/dropout_5/random_uniform/RandomUniform
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2#
!gru_cell/dropout_5/GreaterEqual/yê
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gru_cell/dropout_5/GreaterEqual 
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Cast¦
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/dropout_5/Mul_1
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_1
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2µ
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_2
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stack
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2¿
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_3
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stack
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2¿
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_2
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2¢
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stack
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4¥
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_1
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stack
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2 
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5¥
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_5
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2¾
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_3
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stack
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2¾
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_4
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2¢
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8¥
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_3
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stack
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9¥
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stack
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2Ã
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_5
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stack
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2¥
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¦
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_16719*
condR
while_cond_16718*8
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
runtimeÒ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulå
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul´
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
õ
¥
while_cond_16718
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_16718___redundant_placeholder03
/while_while_cond_16718___redundant_placeholder13
/while_while_cond_16718___redundant_placeholder23
/while_while_cond_16718___redundant_placeholder3
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
ÙS
é
>__inference_gru_layer_call_and_return_conditional_losses_13087

inputs 
gru_cell_12999:`!
gru_cell_13001:	¬` 
gru_cell_13003: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢ gru_cell/StatefulPartitionedCall¢whileD
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
strided_slice_2ß
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_12999gru_cell_13001gru_cell_13003*
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
GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_129982"
 gru_cell/StatefulPartitionedCall
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
while/loop_counterÙ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_12999gru_cell_13001gru_cell_13003*
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
while_body_13011*
condR
while_cond_13010*8
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
runtime¾
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13001*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÑ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13003*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
õ
¥
while_cond_15689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_15689___redundant_placeholder03
/while_while_cond_15689___redundant_placeholder13
/while_while_cond_15689___redundant_placeholder23
/while_while_cond_15689___redundant_placeholder3
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
¦
­
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15039

inputs6
$gru_gru_cell_readvariableop_resource:`9
&gru_gru_cell_readvariableop_1_resource:	¬`8
&gru_gru_cell_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identity¢gru/gru_cell/ReadVariableOp¢gru/gru_cell/ReadVariableOp_1¢gru/gru_cell/ReadVariableOp_2¢gru/gru_cell/ReadVariableOp_3¢gru/gru_cell/ReadVariableOp_4¢gru/gru_cell/ReadVariableOp_5¢gru/gru_cell/ReadVariableOp_6¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢	gru/while¢output/BiasAdd/ReadVariableOp¢output/Tensordot/ReadVariableOpm
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
gru/strided_slice_2
gru/gru_cell/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/Shape
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell/ones_like/Const¹
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/ones_like
gru/gru_cell/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2 
gru/gru_cell/ones_like_1/Shape
gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
gru/gru_cell/ones_like_1/ConstÀ
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/ones_like_1
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell/ReadVariableOp
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell/unstack
gru/gru_cell/mulMulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/mul¡
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/mul_1¡
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/gru_cell/mul_2¦
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru/gru_cell/ReadVariableOp_1
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice/stack_1
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2Í
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice¡
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul¦
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru/gru_cell/ReadVariableOp_2
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_1/stack
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_1/stack_1
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2×
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1©
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_1¦
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru/gru_cell/ReadVariableOp_3
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru/gru_cell/strided_slice_2/stack
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2×
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2©
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_2
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_3/stack_1
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2º
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_3¯
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_4/stack
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_4/stack_1
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2¨
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_4µ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_1
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2$
"gru/gru_cell/strided_slice_5/stack
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2¸
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_5µ
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_2
gru/gru_cell/mul_3Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_3
gru/gru_cell/mul_4Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_4
gru/gru_cell/mul_5Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_5¥
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_4
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_6/stack_1
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2Ö
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6©
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul_3:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_3¥
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_5
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_7/stack
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_7/stack_1
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2Ö
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7©
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_4:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_4
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stack
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_8/stack_1
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2º
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_8µ
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_3
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_9/stack
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_9/stack_1
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2¨
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_9µ
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_4
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid¥
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_1
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid_1¥
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_6
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#gru/gru_cell/strided_slice_10/stack
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2Û
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10ª
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/MatMul_5
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2%
#gru/gru_cell/strided_slice_11/stack
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2½
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_11¶
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/BiasAdd_5
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_6
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Tanh
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_7m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell/sub/x
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/sub
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_8
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_3
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
-gru/TensorArrayUnstack_1/TensorListFromTensorw
gru/zeros_like	ZerosLikegru/gru_cell/add_3:z:0*
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
gru/while/loop_counterÊ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
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
gru_while_body_14834* 
condR
gru_while_cond_14833*M
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
output/BiasAddÖ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulé
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2
	gru/while	gru/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2B
output/Tensordot/ReadVariableOpoutput/Tensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ë	
^
B__inference_masking_layer_call_and_return_conditional_losses_13695

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
ª

Ë
gru_while_cond_15233$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_15233___redundant_placeholder0;
7gru_while_gru_while_cond_15233___redundant_placeholder1;
7gru_while_gru_while_cond_15233___redundant_placeholder2;
7gru_while_gru_while_cond_15233___redundant_placeholder3;
7gru_while_gru_while_cond_15233___redundant_placeholder4
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
ª²
Ì
while_body_15690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	¬`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	¬`:
(while_gru_cell_readvariableop_4_resource: `¢while/gru_cell/ReadVariableOp¢while/gru_cell/ReadVariableOp_1¢while/gru_cell/ReadVariableOp_2¢while/gru_cell/ReadVariableOp_3¢while/gru_cell/ReadVariableOp_4¢while/gru_cell/ReadVariableOp_5¢while/gru_cell/ReadVariableOp_6Ã
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
)while/TensorArrayV2Read/TensorListGetItem 
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
while/gru_cell/ones_like/ConstÁ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/ones_like
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell/ones_like_1/ConstÈ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/ones_like_1§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack·
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul»
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_1»
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_2®
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_1
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2Ù
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul®
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack¡
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1¡
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2ã
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_1®
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack¡
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1¡
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2ã
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_2
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2Æ
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3·
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stack
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2´
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4½
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_1
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stack
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2Ä
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5½
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_5­
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack¡
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1¡
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2â
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6±
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_3­
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack¡
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1¡
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2â
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7±
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_4
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2Æ
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8½
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_3
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stack
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2´
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9½
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_4§
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1­
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/gru_cell/strided_slice_10/stack£
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1£
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ç
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10²
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_5
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stack
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2É
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11¾
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_5¦
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_6¤
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/add_1Ê
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityÝ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ì
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2ù
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3è
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
"

while_body_13343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_gru_cell_13365_0:`)
while_gru_cell_13367_0:	¬`(
while_gru_cell_13369_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_13365:`'
while_gru_cell_13367:	¬`&
while_gru_cell_13369: `¢&while/gru_cell/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem 
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_13365_0while_gru_cell_13367_0while_gru_cell_13369_0*
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
GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_132762(
&while/gru_cell/StatefulPartitionedCalló
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¶
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¼
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4".
while_gru_cell_13365while_gru_cell_13365_0".
while_gru_cell_13367while_gru_cell_13367_0".
while_gru_cell_13369while_gru_cell_13369_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 
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
â
ò
.__inference_GRU_classifier_layer_call_fn_14061	
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_140482
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
ÙÄ
ê
 __inference__wrapped_model_12849	
inputE
3gru_classifier_gru_gru_cell_readvariableop_resource:`H
5gru_classifier_gru_gru_cell_readvariableop_1_resource:	¬`G
5gru_classifier_gru_gru_cell_readvariableop_4_resource: `I
7gru_classifier_output_tensordot_readvariableop_resource: C
5gru_classifier_output_biasadd_readvariableop_resource:
identity¢*GRU_classifier/gru/gru_cell/ReadVariableOp¢,GRU_classifier/gru/gru_cell/ReadVariableOp_1¢,GRU_classifier/gru/gru_cell/ReadVariableOp_2¢,GRU_classifier/gru/gru_cell/ReadVariableOp_3¢,GRU_classifier/gru/gru_cell/ReadVariableOp_4¢,GRU_classifier/gru/gru_cell/ReadVariableOp_5¢,GRU_classifier/gru/gru_cell/ReadVariableOp_6¢GRU_classifier/gru/while¢,GRU_classifier/output/BiasAdd/ReadVariableOp¢.GRU_classifier/output/Tensordot/ReadVariableOp
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
"GRU_classifier/gru/strided_slice_2µ
+GRU_classifier/gru/gru_cell/ones_like/ShapeShape+GRU_classifier/gru/strided_slice_2:output:0*
T0*
_output_shapes
:2-
+GRU_classifier/gru/gru_cell/ones_like/Shape
+GRU_classifier/gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+GRU_classifier/gru/gru_cell/ones_like/Constõ
%GRU_classifier/gru/gru_cell/ones_likeFill4GRU_classifier/gru/gru_cell/ones_like/Shape:output:04GRU_classifier/gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%GRU_classifier/gru/gru_cell/ones_like¯
-GRU_classifier/gru/gru_cell/ones_like_1/ShapeShape!GRU_classifier/gru/zeros:output:0*
T0*
_output_shapes
:2/
-GRU_classifier/gru/gru_cell/ones_like_1/Shape£
-GRU_classifier/gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-GRU_classifier/gru/gru_cell/ones_like_1/Constü
'GRU_classifier/gru/gru_cell/ones_like_1Fill6GRU_classifier/gru/gru_cell/ones_like_1/Shape:output:06GRU_classifier/gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'GRU_classifier/gru/gru_cell/ones_like_1Ì
*GRU_classifier/gru/gru_cell/ReadVariableOpReadVariableOp3gru_classifier_gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02,
*GRU_classifier/gru/gru_cell/ReadVariableOp¾
#GRU_classifier/gru/gru_cell/unstackUnpack2GRU_classifier/gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2%
#GRU_classifier/gru/gru_cell/unstackÙ
GRU_classifier/gru/gru_cell/mulMul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
GRU_classifier/gru/gru_cell/mulÝ
!GRU_classifier/gru/gru_cell/mul_1Mul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!GRU_classifier/gru/gru_cell/mul_1Ý
!GRU_classifier/gru/gru_cell/mul_2Mul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!GRU_classifier/gru/gru_cell/mul_2Ó
,GRU_classifier/gru/gru_cell/ReadVariableOp_1ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_1³
/GRU_classifier/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/GRU_classifier/gru/gru_cell/strided_slice/stack·
1GRU_classifier/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice/stack_1·
1GRU_classifier/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1GRU_classifier/gru/gru_cell/strided_slice/stack_2§
)GRU_classifier/gru/gru_cell/strided_sliceStridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_1:value:08GRU_classifier/gru/gru_cell/strided_slice/stack:output:0:GRU_classifier/gru/gru_cell/strided_slice/stack_1:output:0:GRU_classifier/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2+
)GRU_classifier/gru/gru_cell/strided_sliceÝ
"GRU_classifier/gru/gru_cell/MatMulMatMul#GRU_classifier/gru/gru_cell/mul:z:02GRU_classifier/gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"GRU_classifier/gru/gru_cell/MatMulÓ
,GRU_classifier/gru/gru_cell/ReadVariableOp_2ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_2·
1GRU_classifier/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_1/stack»
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_1»
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_2±
+GRU_classifier/gru/gru_cell/strided_slice_1StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_2:value:0:GRU_classifier/gru/gru_cell/strided_slice_1/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_1/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_1å
$GRU_classifier/gru/gru_cell/MatMul_1MatMul%GRU_classifier/gru/gru_cell/mul_1:z:04GRU_classifier/gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$GRU_classifier/gru/gru_cell/MatMul_1Ó
,GRU_classifier/gru/gru_cell/ReadVariableOp_3ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_3·
1GRU_classifier/gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   23
1GRU_classifier/gru/gru_cell/strided_slice_2/stack»
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_1»
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_2±
+GRU_classifier/gru/gru_cell/strided_slice_2StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_3:value:0:GRU_classifier/gru/gru_cell/strided_slice_2/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_2/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_2å
$GRU_classifier/gru/gru_cell/MatMul_2MatMul%GRU_classifier/gru/gru_cell/mul_2:z:04GRU_classifier/gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$GRU_classifier/gru/gru_cell/MatMul_2°
1GRU_classifier/gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_3/stack´
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_1´
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_2
+GRU_classifier/gru/gru_cell/strided_slice_3StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:0:GRU_classifier/gru/gru_cell/strided_slice_3/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_3/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_3ë
#GRU_classifier/gru/gru_cell/BiasAddBiasAdd,GRU_classifier/gru/gru_cell/MatMul:product:04GRU_classifier/gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell/BiasAdd°
1GRU_classifier/gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_4/stack´
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_1´
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_2
+GRU_classifier/gru/gru_cell/strided_slice_4StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:0:GRU_classifier/gru/gru_cell/strided_slice_4/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_4/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2-
+GRU_classifier/gru/gru_cell/strided_slice_4ñ
%GRU_classifier/gru/gru_cell/BiasAdd_1BiasAdd.GRU_classifier/gru/gru_cell/MatMul_1:product:04GRU_classifier/gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_1°
1GRU_classifier/gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@23
1GRU_classifier/gru/gru_cell/strided_slice_5/stack´
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_1´
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_2
+GRU_classifier/gru/gru_cell/strided_slice_5StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:0:GRU_classifier/gru/gru_cell/strided_slice_5/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_5/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_5ñ
%GRU_classifier/gru/gru_cell/BiasAdd_2BiasAdd.GRU_classifier/gru/gru_cell/MatMul_2:product:04GRU_classifier/gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_2Ô
!GRU_classifier/gru/gru_cell/mul_3Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/mul_3Ô
!GRU_classifier/gru/gru_cell/mul_4Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/mul_4Ô
!GRU_classifier/gru/gru_cell/mul_5Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/mul_5Ò
,GRU_classifier/gru/gru_cell/ReadVariableOp_4ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_4·
1GRU_classifier/gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_6/stack»
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_1»
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_2°
+GRU_classifier/gru/gru_cell/strided_slice_6StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_4:value:0:GRU_classifier/gru/gru_cell/strided_slice_6/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_6/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_6å
$GRU_classifier/gru/gru_cell/MatMul_3MatMul%GRU_classifier/gru/gru_cell/mul_3:z:04GRU_classifier/gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$GRU_classifier/gru/gru_cell/MatMul_3Ò
,GRU_classifier/gru/gru_cell/ReadVariableOp_5ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_5·
1GRU_classifier/gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_7/stack»
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_1»
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_2°
+GRU_classifier/gru/gru_cell/strided_slice_7StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_5:value:0:GRU_classifier/gru/gru_cell/strided_slice_7/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_7/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_7å
$GRU_classifier/gru/gru_cell/MatMul_4MatMul%GRU_classifier/gru/gru_cell/mul_4:z:04GRU_classifier/gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$GRU_classifier/gru/gru_cell/MatMul_4°
1GRU_classifier/gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_8/stack´
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_1´
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_2
+GRU_classifier/gru/gru_cell/strided_slice_8StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:1:GRU_classifier/gru/gru_cell/strided_slice_8/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_8/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_8ñ
%GRU_classifier/gru/gru_cell/BiasAdd_3BiasAdd.GRU_classifier/gru/gru_cell/MatMul_3:product:04GRU_classifier/gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_3°
1GRU_classifier/gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_9/stack´
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_1´
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_2
+GRU_classifier/gru/gru_cell/strided_slice_9StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:1:GRU_classifier/gru/gru_cell/strided_slice_9/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_9/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2-
+GRU_classifier/gru/gru_cell/strided_slice_9ñ
%GRU_classifier/gru/gru_cell/BiasAdd_4BiasAdd.GRU_classifier/gru/gru_cell/MatMul_4:product:04GRU_classifier/gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_4Û
GRU_classifier/gru/gru_cell/addAddV2,GRU_classifier/gru/gru_cell/BiasAdd:output:0.GRU_classifier/gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
GRU_classifier/gru/gru_cell/add¬
#GRU_classifier/gru/gru_cell/SigmoidSigmoid#GRU_classifier/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#GRU_classifier/gru/gru_cell/Sigmoidá
!GRU_classifier/gru/gru_cell/add_1AddV2.GRU_classifier/gru/gru_cell/BiasAdd_1:output:0.GRU_classifier/gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/add_1²
%GRU_classifier/gru/gru_cell/Sigmoid_1Sigmoid%GRU_classifier/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell/Sigmoid_1Ò
,GRU_classifier/gru/gru_cell/ReadVariableOp_6ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_6¹
2GRU_classifier/gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2GRU_classifier/gru/gru_cell/strided_slice_10/stack½
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_1½
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_2µ
,GRU_classifier/gru/gru_cell/strided_slice_10StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_6:value:0;GRU_classifier/gru/gru_cell/strided_slice_10/stack:output:0=GRU_classifier/gru/gru_cell/strided_slice_10/stack_1:output:0=GRU_classifier/gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,GRU_classifier/gru/gru_cell/strided_slice_10æ
$GRU_classifier/gru/gru_cell/MatMul_5MatMul%GRU_classifier/gru/gru_cell/mul_5:z:05GRU_classifier/gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$GRU_classifier/gru/gru_cell/MatMul_5²
2GRU_classifier/gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@24
2GRU_classifier/gru/gru_cell/strided_slice_11/stack¶
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_1¶
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_2
,GRU_classifier/gru/gru_cell/strided_slice_11StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:1;GRU_classifier/gru/gru_cell/strided_slice_11/stack:output:0=GRU_classifier/gru/gru_cell/strided_slice_11/stack_1:output:0=GRU_classifier/gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2.
,GRU_classifier/gru/gru_cell/strided_slice_11ò
%GRU_classifier/gru/gru_cell/BiasAdd_5BiasAdd.GRU_classifier/gru/gru_cell/MatMul_5:product:05GRU_classifier/gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_5Ú
!GRU_classifier/gru/gru_cell/mul_6Mul)GRU_classifier/gru/gru_cell/Sigmoid_1:y:0.GRU_classifier/gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/mul_6Ø
!GRU_classifier/gru/gru_cell/add_2AddV2.GRU_classifier/gru/gru_cell/BiasAdd_2:output:0%GRU_classifier/gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/add_2¥
 GRU_classifier/gru/gru_cell/TanhTanh%GRU_classifier/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 GRU_classifier/gru/gru_cell/TanhË
!GRU_classifier/gru/gru_cell/mul_7Mul'GRU_classifier/gru/gru_cell/Sigmoid:y:0!GRU_classifier/gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/mul_7
!GRU_classifier/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!GRU_classifier/gru/gru_cell/sub/xÐ
GRU_classifier/gru/gru_cell/subSub*GRU_classifier/gru/gru_cell/sub/x:output:0'GRU_classifier/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
GRU_classifier/gru/gru_cell/subÊ
!GRU_classifier/gru/gru_cell/mul_8Mul#GRU_classifier/gru/gru_cell/sub:z:0$GRU_classifier/gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/mul_8Ï
!GRU_classifier/gru/gru_cell/add_3AddV2%GRU_classifier/gru/gru_cell/mul_7:z:0%GRU_classifier/gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!GRU_classifier/gru/gru_cell/add_3µ
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
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor¤
GRU_classifier/gru/zeros_like	ZerosLike%GRU_classifier/gru/gru_cell/add_3:z:0*
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
%GRU_classifier/gru/while/loop_counterº
GRU_classifier/gru/whileWhile.GRU_classifier/gru/while/loop_counter:output:04GRU_classifier/gru/while/maximum_iterations:output:0 GRU_classifier/gru/time:output:0+GRU_classifier/gru/TensorArrayV2_1:handle:0!GRU_classifier/gru/zeros_like:y:0!GRU_classifier/gru/zeros:output:0+GRU_classifier/gru/strided_slice_1:output:0JGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0LGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:03gru_classifier_gru_gru_cell_readvariableop_resource5gru_classifier_gru_gru_cell_readvariableop_1_resource5gru_classifier_gru_gru_cell_readvariableop_4_resource*
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
#GRU_classifier_gru_while_body_12656*/
cond'R%
#GRU_classifier_gru_while_cond_12655*M
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
GRU_classifier/output/BiasAddÉ
IdentityIdentity&GRU_classifier/output/BiasAdd:output:0+^GRU_classifier/gru/gru_cell/ReadVariableOp-^GRU_classifier/gru/gru_cell/ReadVariableOp_1-^GRU_classifier/gru/gru_cell/ReadVariableOp_2-^GRU_classifier/gru/gru_cell/ReadVariableOp_3-^GRU_classifier/gru/gru_cell/ReadVariableOp_4-^GRU_classifier/gru/gru_cell/ReadVariableOp_5-^GRU_classifier/gru/gru_cell/ReadVariableOp_6^GRU_classifier/gru/while-^GRU_classifier/output/BiasAdd/ReadVariableOp/^GRU_classifier/output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2X
*GRU_classifier/gru/gru_cell/ReadVariableOp*GRU_classifier/gru/gru_cell/ReadVariableOp2\
,GRU_classifier/gru/gru_cell/ReadVariableOp_1,GRU_classifier/gru/gru_cell/ReadVariableOp_12\
,GRU_classifier/gru/gru_cell/ReadVariableOp_2,GRU_classifier/gru/gru_cell/ReadVariableOp_22\
,GRU_classifier/gru/gru_cell/ReadVariableOp_3,GRU_classifier/gru/gru_cell/ReadVariableOp_32\
,GRU_classifier/gru/gru_cell/ReadVariableOp_4,GRU_classifier/gru/gru_cell/ReadVariableOp_42\
,GRU_classifier/gru/gru_cell/ReadVariableOp_5,GRU_classifier/gru/gru_cell/ReadVariableOp_52\
,GRU_classifier/gru/gru_cell/ReadVariableOp_6,GRU_classifier/gru/gru_cell/ReadVariableOp_624
GRU_classifier/gru/whileGRU_classifier/gru/while2\
,GRU_classifier/output/BiasAdd/ReadVariableOp,GRU_classifier/output/BiasAdd/ReadVariableOp2`
.GRU_classifier/output/Tensordot/ReadVariableOp.GRU_classifier/output/Tensordot/ReadVariableOp:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
Î
Â
>__inference_gru_layer_call_and_return_conditional_losses_13991

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	¬`4
"gru_cell_readvariableop_4_resource: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell/ReadVariableOp¢gru_cell/ReadVariableOp_1¢gru_cell/ReadVariableOp_2¢gru_cell/ReadVariableOp_3¢gru_cell/ReadVariableOp_4¢gru_cell/ReadVariableOp_5¢gru_cell/ReadVariableOp_6¢whileD
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
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/ones_likev
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like_1/Const°
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/ones_like_1
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_1
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2µ
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_2
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stack
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2¿
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_3
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stack
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2¿
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_2
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2¢
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stack
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4¥
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_1
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stack
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2 
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5¥
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_5
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2¾
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_3
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stack
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2¾
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_4
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2¢
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8¥
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_3
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stack
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9¥
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stack
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2Ã
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_5
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stack
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2¥
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¦
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_13827*
condR
while_cond_13826*8
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
runtimeÒ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulå
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul´
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
'
µ
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14536

inputs
	gru_14511:`
	gru_14513:	¬`
	gru_14515: `
output_14518: 
output_14520:
identity¢gru/StatefulPartitionedCall¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallã
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
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_14511	gru_14513	gru_14515*
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
>__inference_gru_layer_call_and_return_conditional_losses_144752
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14518output_14520*
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
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCall¹
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14513*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÌ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14515*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÁ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
å
ó
.__inference_GRU_classifier_layer_call_fn_14687

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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_145362
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

´
#__inference_gru_layer_call_fn_15526
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
>__inference_gru_layer_call_and_return_conditional_losses_130872
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
®
ç
#__inference_signature_wrapper_14657	
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
 __inference__wrapped_model_128492
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
Ä
¹
__inference_loss_fn_0_17297Q
>gru_gru_cell_kernel_regularizer_square_readvariableop_resource:	¬`
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpî
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>gru_gru_cell_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul¢
IdentityIdentity'gru/gru_cell/kernel/Regularizer/mul:z:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp
÷ù
Ì
while_body_16719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	¬`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	¬`:
(while_gru_cell_readvariableop_4_resource: `¢while/gru_cell/ReadVariableOp¢while/gru_cell/ReadVariableOp_1¢while/gru_cell/ReadVariableOp_2¢while/gru_cell/ReadVariableOp_3¢while/gru_cell/ReadVariableOp_4¢while/gru_cell/ReadVariableOp_5¢while/gru_cell/ReadVariableOp_6Ã
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
)while/TensorArrayV2Read/TensorListGetItem 
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
while/gru_cell/ones_like/ConstÁ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/ones_like
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/Const¼
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Mul
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeþ
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2ìÏ25
3while/gru_cell/dropout/random_uniform/RandomUniform
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2'
%while/gru_cell/dropout/GreaterEqual/yû
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2%
#while/gru_cell/dropout/GreaterEqual­
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Cast·
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout/Mul_1
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/ConstÂ
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_1/Mul
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2øÙ27
5while/gru_cell/dropout_1/random_uniform/RandomUniform
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_1/GreaterEqual/y
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell/dropout_1/GreaterEqual³
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_1/Cast¿
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell/dropout_1/Mul_1
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/ConstÂ
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_2/Mul
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*

seedJ*
seed2¢27
5while/gru_cell/dropout_2/random_uniform/RandomUniform
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_2/GreaterEqual/y
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%while/gru_cell/dropout_2/GreaterEqual³
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/dropout_2/Cast¿
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
while/gru_cell/dropout_2/Mul_1
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 while/gru_cell/ones_like_1/ConstÈ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/ones_like_1
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_3/ConstÃ
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_3/Mul
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Ý27
5while/gru_cell/dropout_3/random_uniform/RandomUniform
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_3/GreaterEqual/y
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_3/GreaterEqual²
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_3/Cast¾
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_3/Mul_1
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_4/ConstÃ
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_4/Mul
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2ô³27
5while/gru_cell/dropout_4/random_uniform/RandomUniform
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_4/GreaterEqual/y
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_4/GreaterEqual²
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_4/Cast¾
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_4/Mul_1
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_5/ConstÃ
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_5/Mul
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2Éù27
5while/gru_cell/dropout_5/random_uniform/RandomUniform
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2)
'while/gru_cell/dropout_5/GreaterEqual/y
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%while/gru_cell/dropout_5/GreaterEqual²
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/dropout_5/Cast¾
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/gru_cell/dropout_5/Mul_1§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¶
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul¼
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_1¼
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/gru_cell/mul_2®
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_1
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2Ù
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul®
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack¡
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1¡
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2ã
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_1®
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack¡
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1¡
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2ã
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_2
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2Æ
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3·
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stack
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2´
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4½
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_1
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stack
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2Ä
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5½
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_5­
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack¡
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1¡
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2â
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6±
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_3­
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack¡
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1¡
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2â
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7±
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_4
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2Æ
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8½
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_3
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stack
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2´
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9½
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_4§
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1­
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/gru_cell/strided_slice_10/stack£
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1£
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ç
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10²
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/MatMul_5
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stack
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2É
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11¾
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/BiasAdd_5¦
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_6¤
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/add_1Ê
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityÝ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ì
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2ù
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3è
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
õ
¥
while_cond_13342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_13342___redundant_placeholder03
/while_while_cond_13342___redundant_placeholder13
/while_while_cond_13342___redundant_placeholder23
/while_while_cond_13342___redundant_placeholder3
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
while_cond_13826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_13826___redundant_placeholder03
/while_while_cond_13826___redundant_placeholder13
/while_while_cond_13826___redundant_placeholder23
/while_while_cond_13826___redundant_placeholder3
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
å
ó
.__inference_GRU_classifier_layer_call_fn_14672

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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_140482
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
while_cond_14262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_14262___redundant_placeholder03
/while_while_cond_14262___redundant_placeholder13
/while_while_cond_14262___redundant_placeholder23
/while_while_cond_14262___redundant_placeholder3
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
µ

C__inference_gru_cell_layer_call_and_return_conditional_losses_12998

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
add_3É
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÜ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÚ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÞ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
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
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
'
´
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14622	
input
	gru_14597:`
	gru_14599:	¬`
	gru_14601: `
output_14604: 
output_14606:
identity¢gru/StatefulPartitionedCall¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢output/StatefulPartitionedCallâ
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
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_14597	gru_14599	gru_14601*
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
>__inference_gru_layer_call_and_return_conditional_losses_144752
gru/StatefulPartitionedCall·
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14604output_14606*
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
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCall¹
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14599*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÌ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14601*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÁ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

_user_specified_nameinput
£»

C__inference_gru_cell_layer_call_and_return_conditional_losses_13276

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
seed2õ­¢2&
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
seed2©x2(
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
seed2¸2(
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
seed2ÑÂ2(
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
dropout_4/ShapeÕ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seedJ*
seed2÷·@2(
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
seed2ºé2(
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
add_3É
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÜ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÚ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÞ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
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
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Ã

C__inference_gru_cell_layer_call_and_return_conditional_losses_17124

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	¬`+
readvariableop_4_resource: `
identity

identity_1¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5¢ReadVariableOp_6¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
add_3É
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÜ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulÚ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÞ

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
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
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
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
íÔ
Ó
gru_while_body_14834$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0c
_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0>
,gru_while_gru_cell_readvariableop_resource_0:`A
.gru_while_gru_cell_readvariableop_1_resource_0:	¬`@
.gru_while_gru_cell_readvariableop_4_resource_0: `
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4
gru_while_identity_5!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensora
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor<
*gru_while_gru_cell_readvariableop_resource:`?
,gru_while_gru_cell_readvariableop_1_resource:	¬`>
,gru_while_gru_cell_readvariableop_4_resource: `¢!gru/while/gru_cell/ReadVariableOp¢#gru/while/gru_cell/ReadVariableOp_1¢#gru/while/gru_cell/ReadVariableOp_2¢#gru/while/gru_cell/ReadVariableOp_3¢#gru/while/gru_cell/ReadVariableOp_4¢#gru/while/gru_cell/ReadVariableOp_5¢#gru/while/gru_cell/ReadVariableOp_6Ë
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
/gru/while/TensorArrayV2Read_1/TensorListGetItem¬
"gru/while/gru_cell/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/Shape
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"gru/while/gru_cell/ones_like/ConstÑ
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/ones_like
$gru/while/gru_cell/ones_like_1/ShapeShapegru_while_placeholder_3*
T0*
_output_shapes
:2&
$gru/while/gru_cell/ones_like_1/Shape
$gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$gru/while/gru_cell/ones_like_1/ConstØ
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
gru/while/gru_cell/ones_like_1³
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02#
!gru/while/gru_cell/ReadVariableOp£
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell/unstackÇ
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/mulË
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/mul_1Ë
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru/while/gru_cell/mul_2º
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1¡
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack¥
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice/stack_1¥
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2ñ
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice¹
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMulº
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2¥
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_1/stack©
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_1/stack_1©
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2û
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1Á
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_1º
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	¬`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3¥
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(gru/while/gru_cell/strided_slice_2/stack©
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1©
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2û
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2Á
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_2
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack¢
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_3/stack_1¢
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2Þ
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_3Ç
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_4/stack¢
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_4/stack_1¢
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2Ì
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_4Í
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_1
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2*
(gru/while/gru_cell/strided_slice_5/stack¢
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1¢
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2Ü
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2$
"gru/while/gru_cell/strided_slice_5Í
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_2¯
gru/while/gru_cell/mul_3Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_3¯
gru/while/gru_cell/mul_4Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_4¯
gru/while/gru_cell/mul_5Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_5¹
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4¥
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack©
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_6/stack_1©
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2ú
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6Á
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_3¹
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5¥
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_7/stack©
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_7/stack_1©
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2ú
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7Á
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_4
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stack¢
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_8/stack_1¢
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2Þ
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_8Í
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_3
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_9/stack¢
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_9/stack_1¢
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2Ì
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_9Í
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_4·
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid½
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_1
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid_1¹
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6§
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)gru/while/gru_cell/strided_slice_10/stack«
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1«
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2ÿ
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10Â
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/MatMul_5 
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2+
)gru/while/gru_cell/strided_slice_11/stack¤
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1¤
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2á
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2%
#gru/while/gru_cell/strided_slice_11Î
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/BiasAdd_5¶
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_6´
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_2
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Tanh¦
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell/sub/x¬
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/sub¦
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_8«
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_3
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
gru/while/Tile¶
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_2*
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
gru/while/Tile_1¼
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_3*
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
gru/while/add_1ò
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1ô
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2¡
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/SelectV2:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_4
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_5"^
,gru_while_gru_cell_readvariableop_1_resource.gru_while_gru_cell_readvariableop_1_resource_0"^
,gru_while_gru_cell_readvariableop_4_resource.gru_while_gru_cell_readvariableop_4_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
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
:: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_6: 
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
Î
Â
>__inference_gru_layer_call_and_return_conditional_losses_16540

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	¬`4
"gru_cell_readvariableop_4_resource: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢gru_cell/ReadVariableOp¢gru_cell/ReadVariableOp_1¢gru_cell/ReadVariableOp_2¢gru_cell/ReadVariableOp_3¢gru_cell/ReadVariableOp_4¢gru_cell/ReadVariableOp_5¢gru_cell/ReadVariableOp_6¢whileD
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
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/ones_likev
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/ones_like_1/Const°
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/ones_like_1
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_1
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2µ
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_2
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stack
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2¿
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype02
gru_cell/ReadVariableOp_3
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stack
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2¿
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	¬ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_2
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2¢
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stack
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4¥
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_1
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stack
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2 
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5¥
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_5
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2¾
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_3
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stack
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2¾
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_4
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2¢
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8¥
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_3
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stack
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9¥
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stack
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2Ã
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/MatMul_5
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stack
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2¥
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¦
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_16376*
condR
while_cond_16375*8
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
runtimeÒ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulå
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul´
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ÙS
é
>__inference_gru_layer_call_and_return_conditional_losses_13419

inputs 
gru_cell_13331:`!
gru_cell_13333:	¬` 
gru_cell_13335: `
identity¢5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp¢?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp¢ gru_cell/StatefulPartitionedCall¢whileD
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
strided_slice_2ß
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_13331gru_cell_13333gru_cell_13335*
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
GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_132762"
 gru_cell/StatefulPartitionedCall
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
while/loop_counterÙ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_13331gru_cell_13333gru_cell_13335*
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
while_body_13343*
condR
while_cond_13342*8
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
runtime¾
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13333*
_output_shapes
:	¬`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpÃ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstÎ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/Sum
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2'
%gru/gru_cell/kernel/Regularizer/mul/xÐ
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulÑ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13335*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpà
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square³
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constö
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/Sum§
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xø
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs"ÌL
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ûÈ
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



 kernel
!recurrent_kernel
"bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
`__call__
*a&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 6}
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
&:$	¬`2gru/gru_cell/kernel
/:- `2gru/gru_cell/recurrent_kernel
#:!`2gru/gru_cell/bias
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
+:)	¬`2Adam/gru/gru_cell/kernel/m
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/m
(:&`2Adam/gru/gru_cell/bias/m
$:" 2Adam/output/kernel/v
:2Adam/output/bias/v
+:)	¬`2Adam/gru/gru_cell/kernel/v
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/v
(:&`2Adam/gru/gru_cell/bias/v
ê2ç
 __inference__wrapped_model_12849Â
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
.__inference_GRU_classifier_layer_call_fn_14061
.__inference_GRU_classifier_layer_call_fn_14672
.__inference_GRU_classifier_layer_call_fn_14687
.__inference_GRU_classifier_layer_call_fn_14564À
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15039
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15487
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14593
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14622À
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
'__inference_masking_layer_call_fn_15492¢
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
B__inference_masking_layer_call_and_return_conditional_losses_15503¢
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
#__inference_gru_layer_call_fn_15526
#__inference_gru_layer_call_fn_15537
#__inference_gru_layer_call_fn_15548
#__inference_gru_layer_call_fn_15559Õ
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
>__inference_gru_layer_call_and_return_conditional_losses_15854
>__inference_gru_layer_call_and_return_conditional_losses_16245
>__inference_gru_layer_call_and_return_conditional_losses_16540
>__inference_gru_layer_call_and_return_conditional_losses_16931Õ
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
&__inference_output_layer_call_fn_16940¢
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
A__inference_output_layer_call_and_return_conditional_losses_16970¢
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
#__inference_signature_wrapper_14657input"
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
2
(__inference_gru_cell_layer_call_fn_16996
(__inference_gru_cell_layer_call_fn_17010¾
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
Î2Ë
C__inference_gru_cell_layer_call_and_return_conditional_losses_17124
C__inference_gru_cell_layer_call_and_return_conditional_losses_17286¾
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
__inference_loss_fn_0_17297
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
__inference_loss_fn_1_17308
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14593" !D¢A
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14622" !D¢A
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15039" !E¢B
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
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15487" !E¢B
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
.__inference_GRU_classifier_layer_call_fn_14061t" !D¢A
:¢7
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
.__inference_GRU_classifier_layer_call_fn_14564t" !D¢A
:¢7
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
.__inference_GRU_classifier_layer_call_fn_14672u" !E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
.__inference_GRU_classifier_layer_call_fn_14687u" !E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
 __inference__wrapped_model_12849" !<¢9
2¢/
-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
ª "<ª9
7
output-*
outputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
C__inference_gru_cell_layer_call_and_return_conditional_losses_17124¸" !]¢Z
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
 
C__inference_gru_cell_layer_call_and_return_conditional_losses_17286¸" !]¢Z
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
 ×
(__inference_gru_cell_layer_call_fn_16996ª" !]¢Z
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
1/0ÿÿÿÿÿÿÿÿÿ ×
(__inference_gru_cell_layer_call_fn_17010ª" !]¢Z
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
>__inference_gru_layer_call_and_return_conditional_losses_15854" !P¢M
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
>__inference_gru_layer_call_and_return_conditional_losses_16245" !P¢M
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
>__inference_gru_layer_call_and_return_conditional_losses_16540" !I¢F
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
>__inference_gru_layer_call_and_return_conditional_losses_16931" !I¢F
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
#__inference_gru_layer_call_fn_15526~" !P¢M
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
#__inference_gru_layer_call_fn_15537~" !P¢M
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
#__inference_gru_layer_call_fn_15548w" !I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
#__inference_gru_layer_call_fn_15559w" !I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :
__inference_loss_fn_0_17297 ¢

¢ 
ª " :
__inference_loss_fn_1_17308!¢

¢ 
ª " º
B__inference_masking_layer_call_and_return_conditional_losses_15503t=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
'__inference_masking_layer_call_fn_15492g=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬»
A__inference_output_layer_call_and_return_conditional_losses_16970v<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
&__inference_output_layer_call_fn_16940i<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
#__inference_signature_wrapper_14657" !E¢B
¢ 
;ª8
6
input-*
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬"<ª9
7
output-*
outputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
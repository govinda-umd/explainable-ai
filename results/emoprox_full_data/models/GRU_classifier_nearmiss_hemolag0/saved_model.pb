п1
С  
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
О
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
і
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
num_elementsintџџџџџџџџџ
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
Ttype"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ІЊ/
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
shape:	Ќ`*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	Ќ`*
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
shape:	Ќ`*+
shared_nameAdam/gru/gru_cell/kernel/m

.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes
:	Ќ`*
dtype0
Є
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
shape:	Ќ`*+
shared_nameAdam/gru/gru_cell/kernel/v

.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes
:	Ќ`*
dtype0
Є
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
dtype0*Э$
valueУ$BР$ BЙ$
й
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
Й
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
!:џџџџџџџџџџџџџџџџџџЌ*
dtype0**
shape!:џџџџџџџџџџџџџџџџџџЌ
Ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kerneloutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*'
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
ы	
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
!__inference__traced_restore_17485ёС.
ы	
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
!:џџџџџџџџџџџџџџџџџџЌ2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Any/reduction_indices
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
mul
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
squeeze_dims

џџџџџџџџџ2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџЌ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
Ё
г
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
.gru_while_gru_cell_readvariableop_1_resource_0:	Ќ`@
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
,gru_while_gru_cell_readvariableop_1_resource:	Ќ`>
,gru_while_gru_cell_readvariableop_4_resource: `Ђ!gru/while/gru_cell/ReadVariableOpЂ#gru/while/gru_cell/ReadVariableOp_1Ђ#gru/while/gru_cell/ReadVariableOp_2Ђ#gru/while/gru_cell/ReadVariableOp_3Ђ#gru/while/gru_cell/ReadVariableOp_4Ђ#gru/while/gru_cell/ReadVariableOp_5Ђ#gru/while/gru_cell/ReadVariableOp_6Ы
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeь
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItemЯ
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeѕ
/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0gru_while_placeholderFgru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
21
/gru/while/TensorArrayV2Read_1/TensorListGetItemЌ
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
"gru/while/gru_cell/ones_like/Constб
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/ones_like
 gru/while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru/while/gru_cell/dropout/ConstЬ
gru/while/gru_cell/dropout/MulMul%gru/while/gru_cell/ones_like:output:0)gru/while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
gru/while/gru_cell/dropout/Mul
 gru/while/gru_cell/dropout/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 gru/while/gru_cell/dropout/Shape
7gru/while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform)gru/while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ьБ29
7gru/while/gru_cell/dropout/random_uniform/RandomUniform
)gru/while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2+
)gru/while/gru_cell/dropout/GreaterEqual/y
'gru/while/gru_cell/dropout/GreaterEqualGreaterEqual@gru/while/gru_cell/dropout/random_uniform/RandomUniform:output:02gru/while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'gru/while/gru_cell/dropout/GreaterEqualЙ
gru/while/gru_cell/dropout/CastCast+gru/while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2!
gru/while/gru_cell/dropout/CastЧ
 gru/while/gru_cell/dropout/Mul_1Mul"gru/while/gru_cell/dropout/Mul:z:0#gru/while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 gru/while/gru_cell/dropout/Mul_1
"gru/while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_1/Constв
 gru/while/gru_cell/dropout_1/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 gru/while/gru_cell/dropout_1/Mul
"gru/while/gru_cell/dropout_1/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_1/Shape
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ГЙm2;
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2-
+gru/while/gru_cell/dropout_1/GreaterEqual/y
)gru/while/gru_cell/dropout_1/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_1/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2+
)gru/while/gru_cell/dropout_1/GreaterEqualП
!gru/while/gru_cell/dropout_1/CastCast-gru/while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2#
!gru/while/gru_cell/dropout_1/CastЯ
"gru/while/gru_cell/dropout_1/Mul_1Mul$gru/while/gru_cell/dropout_1/Mul:z:0%gru/while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"gru/while/gru_cell/dropout_1/Mul_1
"gru/while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_2/Constв
 gru/while/gru_cell/dropout_2/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 gru/while/gru_cell/dropout_2/Mul
"gru/while/gru_cell/dropout_2/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_2/Shape
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ДЁ2;
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2-
+gru/while/gru_cell/dropout_2/GreaterEqual/y
)gru/while/gru_cell/dropout_2/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_2/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2+
)gru/while/gru_cell/dropout_2/GreaterEqualП
!gru/while/gru_cell/dropout_2/CastCast-gru/while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2#
!gru/while/gru_cell/dropout_2/CastЯ
"gru/while/gru_cell/dropout_2/Mul_1Mul$gru/while/gru_cell/dropout_2/Mul:z:0%gru/while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
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
$gru/while/gru_cell/ones_like_1/Constи
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
gru/while/gru_cell/ones_like_1
"gru/while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_3/Constг
 gru/while/gru_cell/dropout_3/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 gru/while/gru_cell/dropout_3/Mul
"gru/while/gru_cell/dropout_3/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_3/Shape
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2лАЩ2;
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2-
+gru/while/gru_cell/dropout_3/GreaterEqual/y
)gru/while/gru_cell/dropout_3/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_3/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)gru/while/gru_cell/dropout_3/GreaterEqualО
!gru/while/gru_cell/dropout_3/CastCast-gru/while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!gru/while/gru_cell/dropout_3/CastЮ
"gru/while/gru_cell/dropout_3/Mul_1Mul$gru/while/gru_cell/dropout_3/Mul:z:0%gru/while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"gru/while/gru_cell/dropout_3/Mul_1
"gru/while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_4/Constг
 gru/while/gru_cell/dropout_4/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 gru/while/gru_cell/dropout_4/Mul
"gru/while/gru_cell/dropout_4/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_4/Shape
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ш2;
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2-
+gru/while/gru_cell/dropout_4/GreaterEqual/y
)gru/while/gru_cell/dropout_4/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_4/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)gru/while/gru_cell/dropout_4/GreaterEqualО
!gru/while/gru_cell/dropout_4/CastCast-gru/while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!gru/while/gru_cell/dropout_4/CastЮ
"gru/while/gru_cell/dropout_4/Mul_1Mul$gru/while/gru_cell/dropout_4/Mul:z:0%gru/while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"gru/while/gru_cell/dropout_4/Mul_1
"gru/while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_5/Constг
 gru/while/gru_cell/dropout_5/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 gru/while/gru_cell/dropout_5/Mul
"gru/while/gru_cell/dropout_5/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_5/Shape
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ЬТ2;
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniform
+gru/while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2-
+gru/while/gru_cell/dropout_5/GreaterEqual/y
)gru/while/gru_cell/dropout_5/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_5/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)gru/while/gru_cell/dropout_5/GreaterEqualО
!gru/while/gru_cell/dropout_5/CastCast-gru/while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!gru/while/gru_cell/dropout_5/CastЮ
"gru/while/gru_cell/dropout_5/Mul_1Mul$gru/while/gru_cell/dropout_5/Mul:z:0%gru/while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"gru/while/gru_cell/dropout_5/Mul_1Г
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02#
!gru/while/gru_cell/ReadVariableOpЃ
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell/unstackЦ
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0$gru/while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/mulЬ
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/mul_1Ь
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/mul_2К
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1Ё
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stackЅ
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice/stack_1Ѕ
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2ё
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_sliceЙ
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMulК
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2Ѕ
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_1/stackЉ
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_1/stack_1Љ
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2ћ
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1С
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_1К
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3Ѕ
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(gru/while/gru_cell/strided_slice_2/stackЉ
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1Љ
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2ћ
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2С
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_2
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stackЂ
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_3/stack_1Ђ
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2о
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_3Ч
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_4/stackЂ
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_4/stack_1Ђ
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2Ь
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_4Э
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_1
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2*
(gru/while/gru_cell/strided_slice_5/stackЂ
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1Ђ
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2м
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2$
"gru/while/gru_cell/strided_slice_5Э
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_2Ў
gru/while/gru_cell/mul_3Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_3Ў
gru/while/gru_cell/mul_4Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_4Ў
gru/while/gru_cell/mul_5Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_5Й
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4Ѕ
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stackЉ
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_6/stack_1Љ
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2њ
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6С
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_3Й
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5Ѕ
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_7/stackЉ
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_7/stack_1Љ
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2њ
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7С
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_4
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stackЂ
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_8/stack_1Ђ
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2о
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_8Э
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_3
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_9/stackЂ
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_9/stack_1Ђ
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2Ь
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_9Э
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_4З
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/SigmoidН
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add_1
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/Sigmoid_1Й
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6Ї
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)gru/while/gru_cell/strided_slice_10/stackЋ
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1Ћ
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2џ
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10Т
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_5 
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2+
)gru/while/gru_cell/strided_slice_11/stackЄ
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1Є
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2с
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2%
#gru/while/gru_cell/strided_slice_11Ю
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_5Ж
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_6Д
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add_2
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/TanhІ
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell/sub/xЌ
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/subІ
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_8Ћ
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add_3
gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile/multiplesЕ
gru/while/TileTile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0!gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
gru/while/TileЖ
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/SelectV2
gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile_1/multiplesЛ
gru/while/Tile_1Tile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0#gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
gru/while/Tile_1М
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/SelectV2_1я
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
gru/while/add_1ђ
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1є
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2Ё
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/SelectV2:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/Identity_4
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_while_identity_5gru/while/Identity_5:output:0"Р
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0"И
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : 2F
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
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :
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
Е
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14048

inputs
	gru_13992:`
	gru_13994:	Ќ`
	gru_13996: `
output_14030: 
output_14032:
identityЂgru/StatefulPartitionedCallЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂoutput/StatefulPartitionedCallу
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCallБ
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_13992	gru_13994	gru_13996*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_139912
gru/StatefulPartitionedCallЗ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14030output_14032*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCallЙ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_13994*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulЬ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_13996*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulС
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
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
while_gru_cell_13035_0:	Ќ`(
while_gru_cell_13037_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_13033:`'
while_gru_cell_13035:	Ќ`&
while_gru_cell_13037: `Ђ&while/gru_cell/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_129982(
&while/gru_cell/StatefulPartitionedCallѓ
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
while/Identity_2Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4".
while_gru_cell_13033while_gru_cell_13033_0".
while_gru_cell_13035while_gru_cell_13035_0".
while_gru_cell_13037while_gru_cell_13037_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2P
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 

Д
#__inference_gru_layer_call_fn_15537
inputs_0
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
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
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
'
Д
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14593	
input
	gru_14568:`
	gru_14570:	Ќ`
	gru_14572: `
output_14575: 
output_14577:
identityЂgru/StatefulPartitionedCallЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂoutput/StatefulPartitionedCallт
masking/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCallБ
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_14568	gru_14570	gru_14572*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_139912
gru/StatefulPartitionedCallЗ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14575output_14577*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCallЙ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14570*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulЬ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14572*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulС
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ

_user_specified_nameinput
ѕ
C
'__inference_masking_layer_call_fn_15492

inputs
identityг
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ* 
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
!:џџџџџџџџџџџџџџџџџџЌ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџЌ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ЊВ
Ь
while_body_13827
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	Ќ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	Ќ`:
(while_gru_cell_readvariableop_4_resource: `Ђwhile/gru_cell/ReadVariableOpЂwhile/gru_cell/ReadVariableOp_1Ђwhile/gru_cell/ReadVariableOp_2Ђwhile/gru_cell/ReadVariableOp_3Ђwhile/gru_cell/ReadVariableOp_4Ђwhile/gru_cell/ReadVariableOp_5Ђwhile/gru_cell/ReadVariableOp_6У
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
while/gru_cell/ones_like/ConstС
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
 while/gru_cell/ones_like_1/ConstШ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/ones_like_1Ї
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
while/gru_cell/unstackЗ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mulЛ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_1Л
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_2Ў
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
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
$while/gru_cell/strided_slice/stack_2й
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
while/gru_cell/strided_sliceЉ
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMulЎ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stackЁ
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1Ё
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2у
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1Б
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMul_1Ў
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stackЁ
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1Ё
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2у
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2Б
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_3/stack_2Ц
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3З
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_4/stack_2Д
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4Н
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_5/stack_2Ф
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5Н
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_6/stackЁ
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1Ё
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2т
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6Б
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_7/stackЁ
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1Ё
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2т
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7Б
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_8/stack_2Ц
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8Н
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_9/stack_2Д
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9Н
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_4Ї
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%while/gru_cell/strided_slice_10/stackЃ
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1Ѓ
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ч
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10В
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
'while/gru_cell/strided_slice_11/stack_2Щ
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11О
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_5І
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_6Є
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_3м
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
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ш
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2>
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
т
ђ
.__inference_GRU_classifier_layer_call_fn_14564	
input
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*'
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
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ

_user_specified_nameinput
м
Ќ
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
=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0:	Ќ`O
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
;gru_classifier_gru_while_gru_cell_readvariableop_1_resource:	Ќ`M
;gru_classifier_gru_while_gru_cell_readvariableop_4_resource: `Ђ0GRU_classifier/gru/while/gru_cell/ReadVariableOpЂ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1Ђ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2Ђ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3Ђ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4Ђ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5Ђ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6щ
JGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2L
JGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeЦ
<GRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0$gru_classifier_gru_while_placeholderSGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02>
<GRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItemэ
LGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2N
LGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeЯ
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0$gru_classifier_gru_while_placeholderUGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
2@
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemй
1GRU_classifier/gru/while/gru_cell/ones_like/ShapeShapeCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1GRU_classifier/gru/while/gru_cell/ones_like/ShapeЋ
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
:џџџџџџџџџЌ2-
+GRU_classifier/gru/while/gru_cell/ones_likeР
3GRU_classifier/gru/while/gru_cell/ones_like_1/ShapeShape&gru_classifier_gru_while_placeholder_3*
T0*
_output_shapes
:25
3GRU_classifier/gru/while/gru_cell/ones_like_1/ShapeЏ
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
:џџџџџџџџџ 2/
-GRU_classifier/gru/while/gru_cell/ones_like_1р
0GRU_classifier/gru/while/gru_cell/ReadVariableOpReadVariableOp;gru_classifier_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype022
0GRU_classifier/gru/while/gru_cell/ReadVariableOpа
)GRU_classifier/gru/while/gru_cell/unstackUnpack8GRU_classifier/gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2+
)GRU_classifier/gru/while/gru_cell/unstack
%GRU_classifier/gru/while/gru_cell/mulMulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%GRU_classifier/gru/while/gru_cell/mul
'GRU_classifier/gru/while/gru_cell/mul_1MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'GRU_classifier/gru/while/gru_cell/mul_1
'GRU_classifier/gru/while/gru_cell/mul_2MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'GRU_classifier/gru/while/gru_cell/mul_2ч
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1П
5GRU_classifier/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5GRU_classifier/gru/while/gru_cell/strided_slice/stackУ
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_1У
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_2Ы
/GRU_classifier/gru/while/gru_cell/strided_sliceStridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_1:value:0>GRU_classifier/gru/while/gru_cell/strided_slice/stack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice/stack_1:output:0@GRU_classifier/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask21
/GRU_classifier/gru/while/gru_cell/strided_sliceѕ
(GRU_classifier/gru/while/gru_cell/MatMulMatMul)GRU_classifier/gru/while/gru_cell/mul:z:08GRU_classifier/gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(GRU_classifier/gru/while/gru_cell/MatMulч
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2У
7GRU_classifier/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_1/stackЧ
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1Ч
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2е
1GRU_classifier/gru/while/gru_cell/strided_slice_1StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_2:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_1/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_1§
*GRU_classifier/gru/while/gru_cell/MatMul_1MatMul+GRU_classifier/gru/while/gru_cell/mul_1:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_1ч
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3У
7GRU_classifier/gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7GRU_classifier/gru/while/gru_cell/strided_slice_2/stackЧ
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1Ч
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2е
1GRU_classifier/gru/while/gru_cell/strided_slice_2StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_3:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_2/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_2§
*GRU_classifier/gru/while/gru_cell/MatMul_2MatMul+GRU_classifier/gru/while/gru_cell/mul_2:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_2М
7GRU_classifier/gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_3/stackР
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1Р
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2И
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
:џџџџџџџџџ 2+
)GRU_classifier/gru/while/gru_cell/BiasAddМ
7GRU_classifier/gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_4/stackР
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1Р
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2І
1GRU_classifier/gru/while/gru_cell/strided_slice_4StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_4/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 23
1GRU_classifier/gru/while/gru_cell/strided_slice_4
+GRU_classifier/gru/while/gru_cell/BiasAdd_1BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_1:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_1М
7GRU_classifier/gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@29
7GRU_classifier/gru/while/gru_cell/strided_slice_5/stackР
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1Р
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2Ж
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
:џџџџџџџџџ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_2ы
'GRU_classifier/gru/while/gru_cell/mul_3Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/mul_3ы
'GRU_classifier/gru/while/gru_cell/mul_4Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/mul_4ы
'GRU_classifier/gru/while/gru_cell/mul_5Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/mul_5ц
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4У
7GRU_classifier/gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_6/stackЧ
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1Ч
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2д
1GRU_classifier/gru/while/gru_cell/strided_slice_6StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_4:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_6/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_6§
*GRU_classifier/gru/while/gru_cell/MatMul_3MatMul+GRU_classifier/gru/while/gru_cell/mul_3:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_3ц
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5У
7GRU_classifier/gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_7/stackЧ
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1Ч
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2д
1GRU_classifier/gru/while/gru_cell/strided_slice_7StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_5:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_7/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_7§
*GRU_classifier/gru/while/gru_cell/MatMul_4MatMul+GRU_classifier/gru/while/gru_cell/mul_4:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_4М
7GRU_classifier/gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_8/stackР
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1Р
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2И
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
:џџџџџџџџџ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_3М
7GRU_classifier/gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_9/stackР
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1Р
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2І
1GRU_classifier/gru/while/gru_cell/strided_slice_9StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1@GRU_classifier/gru/while/gru_cell/strided_slice_9/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 23
1GRU_classifier/gru/while/gru_cell/strided_slice_9
+GRU_classifier/gru/while/gru_cell/BiasAdd_4BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_4:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_4ѓ
%GRU_classifier/gru/while/gru_cell/addAddV22GRU_classifier/gru/while/gru_cell/BiasAdd:output:04GRU_classifier/gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/while/gru_cell/addО
)GRU_classifier/gru/while/gru_cell/SigmoidSigmoid)GRU_classifier/gru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)GRU_classifier/gru/while/gru_cell/Sigmoidљ
'GRU_classifier/gru/while/gru_cell/add_1AddV24GRU_classifier/gru/while/gru_cell/BiasAdd_1:output:04GRU_classifier/gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/add_1Ф
+GRU_classifier/gru/while/gru_cell/Sigmoid_1Sigmoid+GRU_classifier/gru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+GRU_classifier/gru/while/gru_cell/Sigmoid_1ц
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6Х
8GRU_classifier/gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2:
8GRU_classifier/gru/while/gru_cell/strided_slice_10/stackЩ
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1Щ
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2й
2GRU_classifier/gru/while/gru_cell/strided_slice_10StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_6:value:0AGRU_classifier/gru/while/gru_cell/strided_slice_10/stack:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2GRU_classifier/gru/while/gru_cell/strided_slice_10ў
*GRU_classifier/gru/while/gru_cell/MatMul_5MatMul+GRU_classifier/gru/while/gru_cell/mul_5:z:0;GRU_classifier/gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_5О
8GRU_classifier/gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2:
8GRU_classifier/gru/while/gru_cell/strided_slice_11/stackТ
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1Т
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2Л
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
:џџџџџџџџџ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_5ђ
'GRU_classifier/gru/while/gru_cell/mul_6Mul/GRU_classifier/gru/while/gru_cell/Sigmoid_1:y:04GRU_classifier/gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/mul_6№
'GRU_classifier/gru/while/gru_cell/add_2AddV24GRU_classifier/gru/while/gru_cell/BiasAdd_2:output:0+GRU_classifier/gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/add_2З
&GRU_classifier/gru/while/gru_cell/TanhTanh+GRU_classifier/gru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&GRU_classifier/gru/while/gru_cell/Tanhт
'GRU_classifier/gru/while/gru_cell/mul_7Mul-GRU_classifier/gru/while/gru_cell/Sigmoid:y:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/mul_7
'GRU_classifier/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'GRU_classifier/gru/while/gru_cell/sub/xш
%GRU_classifier/gru/while/gru_cell/subSub0GRU_classifier/gru/while/gru_cell/sub/x:output:0-GRU_classifier/gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/while/gru_cell/subт
'GRU_classifier/gru/while/gru_cell/mul_8Mul)GRU_classifier/gru/while/gru_cell/sub:z:0*GRU_classifier/gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/mul_8ч
'GRU_classifier/gru/while/gru_cell/add_3AddV2+GRU_classifier/gru/while/gru_cell/mul_7:z:0+GRU_classifier/gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/while/gru_cell/add_3Ѓ
'GRU_classifier/gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'GRU_classifier/gru/while/Tile/multiplesё
GRU_classifier/gru/while/TileTileEGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem:item:00GRU_classifier/gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
GRU_classifier/gru/while/Tile
!GRU_classifier/gru/while/SelectV2SelectV2&GRU_classifier/gru/while/Tile:output:0+GRU_classifier/gru/while/gru_cell/add_3:z:0&gru_classifier_gru_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/while/SelectV2Ї
)GRU_classifier/gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)GRU_classifier/gru/while/Tile_1/multiplesї
GRU_classifier/gru/while/Tile_1TileEGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem:item:02GRU_classifier/gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2!
GRU_classifier/gru/while/Tile_1
#GRU_classifier/gru/while/SelectV2_1SelectV2(GRU_classifier/gru/while/Tile_1:output:0+GRU_classifier/gru/while/gru_cell/add_3:z:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#GRU_classifier/gru/while/SelectV2_1К
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
GRU_classifier/gru/while/add/yЕ
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
 GRU_classifier/gru/while/add_1/yе
GRU_classifier/gru/while/add_1AddV2>gru_classifier_gru_while_gru_classifier_gru_while_loop_counter)GRU_classifier/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
GRU_classifier/gru/while/add_1
!GRU_classifier/gru/while/IdentityIdentity"GRU_classifier/gru/while/add_1:z:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2#
!GRU_classifier/gru/while/IdentityЎ
#GRU_classifier/gru/while/Identity_1IdentityDgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations1^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_1
#GRU_classifier/gru/while/Identity_2Identity GRU_classifier/gru/while/add:z:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_2З
#GRU_classifier/gru/while/Identity_3IdentityMGRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_3Ѕ
#GRU_classifier/gru/while/Identity_4Identity*GRU_classifier/gru/while/SelectV2:output:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#GRU_classifier/gru/while/Identity_4Ї
#GRU_classifier/gru/while/Identity_5Identity,GRU_classifier/gru/while/SelectV2_1:output:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2%
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
#gru_classifier_gru_while_identity_5,GRU_classifier/gru/while/Identity_5:output:0"ќ
{gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0"є
wgru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensorygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : 2d
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
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Фй
­
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15487

inputs6
$gru_gru_cell_readvariableop_resource:`9
&gru_gru_cell_readvariableop_1_resource:	Ќ`8
&gru_gru_cell_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identityЂgru/gru_cell/ReadVariableOpЂgru/gru_cell/ReadVariableOp_1Ђgru/gru_cell/ReadVariableOp_2Ђgru/gru_cell/ReadVariableOp_3Ђgru/gru_cell/ReadVariableOp_4Ђgru/gru_cell/ReadVariableOp_5Ђgru/gru_cell/ReadVariableOp_6Ђ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂ	gru/whileЂoutput/BiasAdd/ReadVariableOpЂoutput/Tensordot/ReadVariableOpm
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
!:џџџџџџџџџџџџџџџџџџЌ2
masking/NotEqual
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
masking/Any/reduction_indicesІ
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
masking/Any
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
masking/mul
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
squeeze_dims

џџџџџџџџџ2
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
gru/strided_slice/stack_2њ
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
џџџџџџџџџ2
gru/ExpandDims/dimЄ
gru/ExpandDims
ExpandDimsmasking/Squeeze:output:0gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru/ExpandDims
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/permІ
gru/transpose_1	Transposegru/ExpandDims:output:0gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru/transpose_1
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
gru/TensorArrayV2/element_shapeТ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2Ч
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2;
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
:џџџџџџџџџЌ*
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
gru/gru_cell/ones_like/ConstЙ
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/ones_like}
gru/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout/ConstД
gru/gru_cell/dropout/MulMulgru/gru_cell/ones_like:output:0#gru/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout/Mul
gru/gru_cell/dropout/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout/Shapeј
1gru/gru_cell/dropout/random_uniform/RandomUniformRandomUniform#gru/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ТЮ23
1gru/gru_cell/dropout/random_uniform/RandomUniform
#gru/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2%
#gru/gru_cell/dropout/GreaterEqual/yѓ
!gru/gru_cell/dropout/GreaterEqualGreaterEqual:gru/gru_cell/dropout/random_uniform/RandomUniform:output:0,gru/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!gru/gru_cell/dropout/GreaterEqualЇ
gru/gru_cell/dropout/CastCast%gru/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout/CastЏ
gru/gru_cell/dropout/Mul_1Mulgru/gru_cell/dropout/Mul:z:0gru/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout/Mul_1
gru/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_1/ConstК
gru/gru_cell/dropout_1/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout_1/Mul
gru/gru_cell/dropout_1/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_1/Shapeў
3gru/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2зЧ25
3gru/gru_cell/dropout_1/random_uniform/RandomUniform
%gru/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%gru/gru_cell/dropout_1/GreaterEqual/yћ
#gru/gru_cell/dropout_1/GreaterEqualGreaterEqual<gru/gru_cell/dropout_1/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#gru/gru_cell/dropout_1/GreaterEqual­
gru/gru_cell/dropout_1/CastCast'gru/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout_1/CastЗ
gru/gru_cell/dropout_1/Mul_1Mulgru/gru_cell/dropout_1/Mul:z:0gru/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout_1/Mul_1
gru/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_2/ConstК
gru/gru_cell/dropout_2/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout_2/Mul
gru/gru_cell/dropout_2/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_2/Shapeў
3gru/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2Прѕ25
3gru/gru_cell/dropout_2/random_uniform/RandomUniform
%gru/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%gru/gru_cell/dropout_2/GreaterEqual/yћ
#gru/gru_cell/dropout_2/GreaterEqualGreaterEqual<gru/gru_cell/dropout_2/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#gru/gru_cell/dropout_2/GreaterEqual­
gru/gru_cell/dropout_2/CastCast'gru/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/dropout_2/CastЗ
gru/gru_cell/dropout_2/Mul_1Mulgru/gru_cell/dropout_2/Mul:z:0gru/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru/gru_cell/ones_like_1/ConstР
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/ones_like_1
gru/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_3/ConstЛ
gru/gru_cell/dropout_3/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_3/Mul
gru/gru_cell/dropout_3/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_3/Shape§
3gru/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2н25
3gru/gru_cell/dropout_3/random_uniform/RandomUniform
%gru/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%gru/gru_cell/dropout_3/GreaterEqual/yњ
#gru/gru_cell/dropout_3/GreaterEqualGreaterEqual<gru/gru_cell/dropout_3/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#gru/gru_cell/dropout_3/GreaterEqualЌ
gru/gru_cell/dropout_3/CastCast'gru/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_3/CastЖ
gru/gru_cell/dropout_3/Mul_1Mulgru/gru_cell/dropout_3/Mul:z:0gru/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_3/Mul_1
gru/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_4/ConstЛ
gru/gru_cell/dropout_4/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_4/Mul
gru/gru_cell/dropout_4/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_4/Shapeќ
3gru/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ГМ25
3gru/gru_cell/dropout_4/random_uniform/RandomUniform
%gru/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%gru/gru_cell/dropout_4/GreaterEqual/yњ
#gru/gru_cell/dropout_4/GreaterEqualGreaterEqual<gru/gru_cell/dropout_4/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#gru/gru_cell/dropout_4/GreaterEqualЌ
gru/gru_cell/dropout_4/CastCast'gru/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_4/CastЖ
gru/gru_cell/dropout_4/Mul_1Mulgru/gru_cell/dropout_4/Mul:z:0gru/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_4/Mul_1
gru/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_5/ConstЛ
gru/gru_cell/dropout_5/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_5/Mul
gru/gru_cell/dropout_5/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_5/Shapeќ
3gru/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2т(25
3gru/gru_cell/dropout_5/random_uniform/RandomUniform
%gru/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%gru/gru_cell/dropout_5/GreaterEqual/yњ
#gru/gru_cell/dropout_5/GreaterEqualGreaterEqual<gru/gru_cell/dropout_5/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#gru/gru_cell/dropout_5/GreaterEqualЌ
gru/gru_cell/dropout_5/CastCast'gru/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/dropout_5/CastЖ
gru/gru_cell/dropout_5/Mul_1Mulgru/gru_cell/dropout_5/Mul:z:0gru/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru/gru_cell/mulЂ
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/mul_1Ђ
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/mul_2І
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
"gru/gru_cell/strided_slice/stack_2Э
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru/gru_cell/strided_sliceЁ
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/MatMulІ
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
$gru/gru_cell/strided_slice_1/stack_2з
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1Љ
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/MatMul_1І
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
$gru/gru_cell/strided_slice_2/stack_2з
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2Љ
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_3/stack_2К
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_3Џ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_4/stack_2Ј
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_4Е
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_5/stack_2И
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_5Е
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/BiasAdd_2
gru/gru_cell/mul_3Mulgru/zeros:output:0 gru/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_3
gru/gru_cell/mul_4Mulgru/zeros:output:0 gru/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_4
gru/gru_cell/mul_5Mulgru/zeros:output:0 gru/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_5Ѕ
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
$gru/gru_cell/strided_slice_6/stack_2ж
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6Љ
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul_3:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/MatMul_3Ѕ
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
$gru/gru_cell/strided_slice_7/stack_2ж
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7Љ
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_4:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_8/stack_2К
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_8Е
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_9/stack_2Ј
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_9Е
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/BiasAdd_4
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/SigmoidЅ
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add_1
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/Sigmoid_1Ѕ
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
%gru/gru_cell/strided_slice_10/stack_2л
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10Њ
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%gru/gru_cell/strided_slice_11/stack_2Н
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_11Ж
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/BiasAdd_5
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_6
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/Tanh
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru/gru_cell/sub
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_8
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add_3
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2#
!gru/TensorArrayV2_1/element_shapeШ
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
џџџџџџџџџ2#
!gru/TensorArrayV2_2/element_shapeШ
gru/TensorArrayV2_2TensorListReserve*gru/TensorArrayV2_2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
gru/TensorArrayV2_2Ы
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
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
:џџџџџџџџџ 2
gru/zeros_like
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counterЪ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
gru_while_body_15234* 
condR
gru_while_cond_15233*M
output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *
parallel_iterations 2
	gru/whileН
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4gru/TensorArrayV2Stack/TensorListStack/element_shape
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
gru/strided_slice_3/stack_2В
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
gru/strided_slice_3
gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_2/permО
gru/transpose_2	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
gru/transpose_2n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtimeЋ
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
output/Tensordot/GatherV2/axisє
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
 output/Tensordot/GatherV2_1/axisњ
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
output/Tensordot/Const_1Є
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
output/Tensordot/concat/axisг
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concatЈ
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stackЛ
output/Tensordot/transpose	Transposegru/transpose_2:y:0 output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
output/Tensordot/transposeЛ
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
output/Tensordot/ReshapeК
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
output/Tensordot/concat_1/axisр
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1Е
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
output/TensordotЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЌ
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
output/BiasAddж
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulщ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2:
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
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
їљ
Ь
while_body_16033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	Ќ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	Ќ`:
(while_gru_cell_readvariableop_4_resource: `Ђwhile/gru_cell/ReadVariableOpЂwhile/gru_cell/ReadVariableOp_1Ђwhile/gru_cell/ReadVariableOp_2Ђwhile/gru_cell/ReadVariableOp_3Ђwhile/gru_cell/ReadVariableOp_4Ђwhile/gru_cell/ReadVariableOp_5Ђwhile/gru_cell/ReadVariableOp_6У
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
while/gru_cell/ones_like/ConstС
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/ones_like
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/ConstМ
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/Mul
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeў
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2щЅЇ25
3while/gru_cell/dropout/random_uniform/RandomUniform
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%while/gru_cell/dropout/GreaterEqual/yћ
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#while/gru_cell/dropout/GreaterEqual­
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/CastЗ
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/Mul_1
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/ConstТ
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_1/Mul
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2щЫл27
5while/gru_cell/dropout_1/random_uniform/RandomUniform
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_1/GreaterEqual/y
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%while/gru_cell/dropout_1/GreaterEqualГ
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_1/CastП
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
while/gru_cell/dropout_1/Mul_1
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/ConstТ
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_2/Mul
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2дЛо27
5while/gru_cell/dropout_2/random_uniform/RandomUniform
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_2/GreaterEqual/y
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%while/gru_cell/dropout_2/GreaterEqualГ
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_2/CastП
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
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
 while/gru_cell/ones_like_1/ConstШ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/ones_like_1
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_3/ConstУ
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_3/Mul
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ялЮ27
5while/gru_cell/dropout_3/random_uniform/RandomUniform
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_3/GreaterEqual/y
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_3/GreaterEqualВ
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_3/CastО
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_3/Mul_1
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_4/ConstУ
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_4/Mul
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2њрй27
5while/gru_cell/dropout_4/random_uniform/RandomUniform
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_4/GreaterEqual/y
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_4/GreaterEqualВ
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_4/CastО
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_4/Mul_1
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_5/ConstУ
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_5/Mul
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2єыў27
5while/gru_cell/dropout_5/random_uniform/RandomUniform
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_5/GreaterEqual/y
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_5/GreaterEqualВ
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_5/CastО
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_5/Mul_1Ї
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
while/gru_cell/unstackЖ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mulМ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_1М
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_2Ў
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
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
$while/gru_cell/strided_slice/stack_2й
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
while/gru_cell/strided_sliceЉ
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMulЎ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stackЁ
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1Ё
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2у
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1Б
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMul_1Ў
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stackЁ
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1Ё
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2у
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2Б
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_3/stack_2Ц
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3З
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_4/stack_2Д
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4Н
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_5/stack_2Ф
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5Н
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_6/stackЁ
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1Ё
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2т
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6Б
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_7/stackЁ
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1Ё
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2т
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7Б
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_8/stack_2Ц
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8Н
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_9/stack_2Д
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9Н
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_4Ї
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%while/gru_cell/strided_slice_10/stackЃ
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1Ѓ
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ч
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10В
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
'while/gru_cell/strided_slice_11/stack_2Щ
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11О
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_5І
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_6Є
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_3м
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
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ш
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2>
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Ю

&__inference_output_layer_call_fn_16940

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
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
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ѕљ
Ь
while_body_14263
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	Ќ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	Ќ`:
(while_gru_cell_readvariableop_4_resource: `Ђwhile/gru_cell/ReadVariableOpЂwhile/gru_cell/ReadVariableOp_1Ђwhile/gru_cell/ReadVariableOp_2Ђwhile/gru_cell/ReadVariableOp_3Ђwhile/gru_cell/ReadVariableOp_4Ђwhile/gru_cell/ReadVariableOp_5Ђwhile/gru_cell/ReadVariableOp_6У
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
while/gru_cell/ones_like/ConstС
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/ones_like
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/ConstМ
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/Mul
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeў
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2Раи25
3while/gru_cell/dropout/random_uniform/RandomUniform
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%while/gru_cell/dropout/GreaterEqual/yћ
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#while/gru_cell/dropout/GreaterEqual­
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/CastЗ
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/Mul_1
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/ConstТ
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_1/Mul
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ЬР 27
5while/gru_cell/dropout_1/random_uniform/RandomUniform
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_1/GreaterEqual/y
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%while/gru_cell/dropout_1/GreaterEqualГ
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_1/CastП
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
while/gru_cell/dropout_1/Mul_1
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/ConstТ
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_2/Mul
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ћоз27
5while/gru_cell/dropout_2/random_uniform/RandomUniform
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_2/GreaterEqual/y
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%while/gru_cell/dropout_2/GreaterEqualГ
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_2/CastП
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
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
 while/gru_cell/ones_like_1/ConstШ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/ones_like_1
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_3/ConstУ
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_3/Mul
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ЮЕЎ27
5while/gru_cell/dropout_3/random_uniform/RandomUniform
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_3/GreaterEqual/y
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_3/GreaterEqualВ
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_3/CastО
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_3/Mul_1
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_4/ConstУ
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_4/Mul
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed227
5while/gru_cell/dropout_4/random_uniform/RandomUniform
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_4/GreaterEqual/y
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_4/GreaterEqualВ
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_4/CastО
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_4/Mul_1
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_5/ConstУ
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_5/Mul
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2фЇ27
5while/gru_cell/dropout_5/random_uniform/RandomUniform
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_5/GreaterEqual/y
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_5/GreaterEqualВ
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_5/CastО
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_5/Mul_1Ї
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
while/gru_cell/unstackЖ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mulМ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_1М
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_2Ў
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
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
$while/gru_cell/strided_slice/stack_2й
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
while/gru_cell/strided_sliceЉ
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMulЎ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stackЁ
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1Ё
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2у
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1Б
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMul_1Ў
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stackЁ
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1Ё
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2у
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2Б
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_3/stack_2Ц
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3З
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_4/stack_2Д
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4Н
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_5/stack_2Ф
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5Н
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_6/stackЁ
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1Ё
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2т
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6Б
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_7/stackЁ
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1Ё
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2т
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7Б
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_8/stack_2Ц
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8Н
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_9/stack_2Д
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9Н
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_4Ї
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%while/gru_cell/strided_slice_10/stackЃ
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1Ѓ
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ч
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10В
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
'while/gru_cell/strided_slice_11/stack_2Щ
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11О
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_5І
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_6Є
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_3м
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
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ш
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2>
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Њ

Ы
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
D: : : : :џџџџџџџџџ :џџџџџџџџџ : :::::: 
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
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Т 
ј
A__inference_output_layer_call_and_return_conditional_losses_14029

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
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
Tensordot/GatherV2/axisб
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
Tensordot/GatherV2_1/axisз
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
Tensordot/concat/axisА
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
 :џџџџџџџџџџџџџџџџџџ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2	
BiasAddЅ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ЊВ
Ь
while_body_16376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	Ќ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	Ќ`:
(while_gru_cell_readvariableop_4_resource: `Ђwhile/gru_cell/ReadVariableOpЂwhile/gru_cell/ReadVariableOp_1Ђwhile/gru_cell/ReadVariableOp_2Ђwhile/gru_cell/ReadVariableOp_3Ђwhile/gru_cell/ReadVariableOp_4Ђwhile/gru_cell/ReadVariableOp_5Ђwhile/gru_cell/ReadVariableOp_6У
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
while/gru_cell/ones_like/ConstС
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
 while/gru_cell/ones_like_1/ConstШ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/ones_like_1Ї
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
while/gru_cell/unstackЗ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mulЛ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_1Л
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_2Ў
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
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
$while/gru_cell/strided_slice/stack_2й
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
while/gru_cell/strided_sliceЉ
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMulЎ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stackЁ
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1Ё
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2у
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1Б
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMul_1Ў
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stackЁ
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1Ё
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2у
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2Б
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_3/stack_2Ц
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3З
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_4/stack_2Д
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4Н
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_5/stack_2Ф
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5Н
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_6/stackЁ
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1Ё
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2т
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6Б
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_7/stackЁ
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1Ё
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2т
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7Б
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_8/stack_2Ц
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8Н
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_9/stack_2Д
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9Н
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_4Ї
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%while/gru_cell/strided_slice_10/stackЃ
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1Ѓ
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ч
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10В
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
'while/gru_cell/strided_slice_11/stack_2Щ
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11О
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_5І
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_6Є
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_3м
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
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ш
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2>
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
т
Ф
>__inference_gru_layer_call_and_return_conditional_losses_16245
inputs_02
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	Ќ`4
"gru_cell_readvariableop_4_resource: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂgru_cell/ReadVariableOpЂgru_cell/ReadVariableOp_1Ђgru_cell/ReadVariableOp_2Ђgru_cell/ReadVariableOp_3Ђgru_cell/ReadVariableOp_4Ђgru_cell/ReadVariableOp_5Ђgru_cell/ReadVariableOp_6ЂwhileF
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
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
gru_cell/ones_like/ConstЉ
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/ConstЄ
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeь
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2зНЬ2/
-gru_cell/dropout/random_uniform/RandomUniform
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2!
gru_cell/dropout/GreaterEqual/yу
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/GreaterEqual
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Cast
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/ConstЊ
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeђ
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ѓє21
/gru_cell/dropout_1/random_uniform/RandomUniform
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_1/GreaterEqual/yы
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
gru_cell/dropout_1/GreaterEqualЁ
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/CastЇ
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/ConstЊ
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeђ
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2Пф21
/gru_cell/dropout_2/random_uniform/RandomUniform
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_2/GreaterEqual/yы
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
gru_cell/dropout_2/GreaterEqualЁ
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_2/CastЇ
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru_cell/ones_like_1/ConstА
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_3/ConstЋ
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/Mul
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeё
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2Џое21
/gru_cell/dropout_3/random_uniform/RandomUniform
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_3/GreaterEqual/yъ
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_3/GreaterEqual 
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/CastІ
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_4/ConstЋ
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/Mul
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeё
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2У21
/gru_cell/dropout_4/random_uniform/RandomUniform
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_4/GreaterEqual/yъ
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_4/GreaterEqual 
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/CastІ
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_5/ConstЋ
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_5/Mul
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeё
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed221
/gru_cell/dropout_5/random_uniform/RandomUniform
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_5/GreaterEqual/yъ
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_5/GreaterEqual 
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_5/CastІ
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
gru_cell/strided_slice/stack_2Е
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_1/stack_2П
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_2/stack_2П
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_3/stack_2Ђ
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
:џџџџџџџџџ 2
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
gru_cell/strided_slice_4Ѕ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_5Ѕ
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_6/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_7/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_8/stack_2Ђ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8Ѕ
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_9Ѕ
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_10/stack_2У
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
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_11/stack_2Ѕ
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11І
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
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
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_16033*
condR
while_cond_16032*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeв
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulх
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulД
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
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
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
к
Т
>__inference_gru_layer_call_and_return_conditional_losses_14475

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	Ќ`4
"gru_cell_readvariableop_4_resource: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂgru_cell/ReadVariableOpЂgru_cell/ReadVariableOp_1Ђgru_cell/ReadVariableOp_2Ђgru_cell/ReadVariableOp_3Ђgru_cell/ReadVariableOp_4Ђgru_cell/ReadVariableOp_5Ђgru_cell/ReadVariableOp_6ЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
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
gru_cell/ones_like/ConstЉ
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/ConstЄ
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeь
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2цЕ2/
-gru_cell/dropout/random_uniform/RandomUniform
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2!
gru_cell/dropout/GreaterEqual/yу
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/GreaterEqual
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Cast
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/ConstЊ
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeђ
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2єпљ21
/gru_cell/dropout_1/random_uniform/RandomUniform
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_1/GreaterEqual/yы
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
gru_cell/dropout_1/GreaterEqualЁ
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/CastЇ
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/ConstЊ
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeђ
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2 ш21
/gru_cell/dropout_2/random_uniform/RandomUniform
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_2/GreaterEqual/yы
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
gru_cell/dropout_2/GreaterEqualЁ
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_2/CastЇ
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru_cell/ones_like_1/ConstА
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_3/ConstЋ
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/Mul
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeё
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2џЛ21
/gru_cell/dropout_3/random_uniform/RandomUniform
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_3/GreaterEqual/yъ
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_3/GreaterEqual 
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/CastІ
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_4/ConstЋ
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/Mul
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeё
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2бє­21
/gru_cell/dropout_4/random_uniform/RandomUniform
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_4/GreaterEqual/yъ
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_4/GreaterEqual 
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/CastІ
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_5/ConstЋ
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_5/Mul
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeё
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2Ушќ21
/gru_cell/dropout_5/random_uniform/RandomUniform
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_5/GreaterEqual/yъ
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_5/GreaterEqual 
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_5/CastІ
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
gru_cell/strided_slice/stack_2Е
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_1/stack_2П
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_2/stack_2П
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_3/stack_2Ђ
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
:џџџџџџџџџ 2
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
gru_cell/strided_slice_4Ѕ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_5Ѕ
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_6/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_7/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_8/stack_2Ђ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8Ѕ
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_9Ѕ
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_10/stack_2У
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
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_11/stack_2Ѕ
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11І
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
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
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_14263*
condR
while_cond_14262*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeв
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulх
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulД
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
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
!:џџџџџџџџџџџџџџџџџџЌ
 
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
Я
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
D: : : : :џџџџџџџџџ :џџџџџџџџџ : :::::: 
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
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ВЛ

C__inference_gru_cell_layer_call_and_return_conditional_losses_17286

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	Ќ`+
readvariableop_4_resource: `
identity

identity_1ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4ЂReadVariableOp_5ЂReadVariableOp_6Ђ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
:џџџџџџџџџЌ2
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
:џџџџџџџџџЌ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeб
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2њ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
:џџџџџџџџџЌ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeз
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2лЃз2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_1/GreaterEqual/yЧ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
:џџџџџџџџџЌ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeж
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2[2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_2/GreaterEqual/yЧ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeж
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2Єз2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeж
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed22(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeж
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2рН2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
strided_slice/stack_2џ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

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
strided_slice_3/stack_2ь
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
:џџџџџџџџџ 2	
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
strided_slice_4/stack_2к
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
strided_slice_5/stack_2ъ
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
:џџџџџџџџџ 2
	BiasAdd_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
:џџџџџџџџџ 2

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
strided_slice_8/stack_2ь
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
:џџџџџџџџџ 2
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
strided_slice_9/stack_2к
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
strided_slice_11/stack_2я
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
:џџџџџџџџџ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Щ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulм
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulк
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџ : : : 2 
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
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0
ћ
В
#__inference_gru_layer_call_fn_15548

inputs
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
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
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Т 
ј
A__inference_output_layer_call_and_return_conditional_losses_16970

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
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
Tensordot/GatherV2/axisб
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
Tensordot/GatherV2_1/axisз
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
Tensordot/concat/axisА
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
 :џџџџџџџџџџџџџџџџџџ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2	
BiasAddЅ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ћ
В
#__inference_gru_layer_call_fn_15559

inputs
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
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
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Ќ

е
(__inference_gru_cell_layer_call_fn_16996

inputs
states_0
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
identity

identity_1ЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
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
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0
њ8
Ї

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

identity_1ЂMergeV2Checkpoints
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesК
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЎ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*И
_input_shapesІ
Ѓ: : :: : : : : :	Ќ`: `:`: : : : : ::	Ќ`: `:`: ::	Ќ`: `:`: 2(
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
:	Ќ`:$	 

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
:	Ќ`:$ 

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
:	Ќ`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:

_output_shapes
: 
ЂЮ
Ф
>__inference_gru_layer_call_and_return_conditional_losses_15854
inputs_02
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	Ќ`4
"gru_cell_readvariableop_4_resource: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂgru_cell/ReadVariableOpЂgru_cell/ReadVariableOp_1Ђgru_cell/ReadVariableOp_2Ђgru_cell/ReadVariableOp_3Ђgru_cell/ReadVariableOp_4Ђgru_cell/ReadVariableOp_5Ђgru_cell/ReadVariableOp_6ЂwhileF
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
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
gru_cell/ones_like/ConstЉ
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru_cell/ones_like_1/ConstА
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
gru_cell/strided_slice/stack_2Е
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_1/stack_2П
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_2/stack_2П
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_3/stack_2Ђ
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
:џџџџџџџџџ 2
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
gru_cell/strided_slice_4Ѕ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_5Ѕ
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_6/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_7/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_8/stack_2Ђ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8Ѕ
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_9Ѕ
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_10/stack_2У
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
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_11/stack_2Ѕ
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11І
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
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
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_15690*
condR
while_cond_15689*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeв
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulх
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulД
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
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
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
вh
г
!__inference__traced_restore_17485
file_prefix0
assignvariableop_output_kernel: ,
assignvariableop_1_output_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: 9
&assignvariableop_7_gru_gru_cell_kernel:	Ќ`B
0assignvariableop_8_gru_gru_cell_recurrent_kernel: `6
$assignvariableop_9_gru_gru_cell_bias:`#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: :
(assignvariableop_14_adam_output_kernel_m: 4
&assignvariableop_15_adam_output_bias_m:A
.assignvariableop_16_adam_gru_gru_cell_kernel_m:	Ќ`J
8assignvariableop_17_adam_gru_gru_cell_recurrent_kernel_m: `>
,assignvariableop_18_adam_gru_gru_cell_bias_m:`:
(assignvariableop_19_adam_output_kernel_v: 4
&assignvariableop_20_adam_output_bias_v:A
.assignvariableop_21_adam_gru_gru_cell_kernel_v:	Ќ`J
8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_v: `>
,assignvariableop_23_adam_gru_gru_cell_bias_v:`
identity_25ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesР
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЈ
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

Identity_1Ѓ
AssignVariableOp_1AssignVariableOpassignvariableop_1_output_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2Ё
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѓ
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѓ
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ђ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Њ
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ћ
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gru_gru_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Е
AssignVariableOp_8AssignVariableOp0assignvariableop_8_gru_gru_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Љ
AssignVariableOp_9AssignVariableOp$assignvariableop_9_gru_gru_cell_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ё
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ѓ
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_output_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ў
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_output_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ж
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_gru_gru_cell_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Р
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_gru_gru_cell_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Д
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_gru_gru_cell_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19А
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_output_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ў
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_output_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ж
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_gru_gru_cell_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Р
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Д
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_gru_gru_cell_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpю
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24с
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
В
Ь
__inference_loss_fn_1_17308Z
Hgru_gru_cell_recurrent_kernel_regularizer_square_readvariableop_resource: `
identityЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpHgru_gru_cell_recurrent_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЖ
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
Ќ

е
(__inference_gru_cell_layer_call_fn_17010

inputs
states_0
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
identity

identity_1ЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
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
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0
и
Т
>__inference_gru_layer_call_and_return_conditional_losses_16931

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	Ќ`4
"gru_cell_readvariableop_4_resource: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂgru_cell/ReadVariableOpЂgru_cell/ReadVariableOp_1Ђgru_cell/ReadVariableOp_2Ђgru_cell/ReadVariableOp_3Ђgru_cell/ReadVariableOp_4Ђgru_cell/ReadVariableOp_5Ђgru_cell/ReadVariableOp_6ЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
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
gru_cell/ones_like/ConstЉ
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/ConstЄ
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeы
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ѓЁp2/
-gru_cell/dropout/random_uniform/RandomUniform
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2!
gru_cell/dropout/GreaterEqual/yу
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/GreaterEqual
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Cast
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/ConstЊ
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeђ
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ЖШљ21
/gru_cell/dropout_1/random_uniform/RandomUniform
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_1/GreaterEqual/yы
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
gru_cell/dropout_1/GreaterEqualЁ
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/CastЇ
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/ConstЊ
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeђ
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ёБ­21
/gru_cell/dropout_2/random_uniform/RandomUniform
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_2/GreaterEqual/yы
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
gru_cell/dropout_2/GreaterEqualЁ
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/dropout_2/CastЇ
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru_cell/ones_like_1/ConstА
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_3/ConstЋ
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/Mul
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeё
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ЙЯ21
/gru_cell/dropout_3/random_uniform/RandomUniform
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_3/GreaterEqual/yъ
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_3/GreaterEqual 
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/CastІ
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_4/ConstЋ
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/Mul
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeё
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2ЯЇ21
/gru_cell/dropout_4/random_uniform/RandomUniform
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_4/GreaterEqual/yъ
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_4/GreaterEqual 
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/CastІ
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_5/ConstЋ
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_5/Mul
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shape№
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2]21
/gru_cell/dropout_5/random_uniform/RandomUniform
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2#
!gru_cell/dropout_5/GreaterEqual/yъ
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
gru_cell/dropout_5/GreaterEqual 
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/dropout_5/CastІ
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
gru_cell/strided_slice/stack_2Е
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_1/stack_2П
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_2/stack_2П
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_3/stack_2Ђ
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
:џџџџџџџџџ 2
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
gru_cell/strided_slice_4Ѕ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_5Ѕ
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_6/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_7/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_8/stack_2Ђ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8Ѕ
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_9Ѕ
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_10/stack_2У
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
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_11/stack_2Ѕ
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11І
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
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
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_16719*
condR
while_cond_16718*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeв
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulх
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulД
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
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
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
йS
щ
>__inference_gru_layer_call_and_return_conditional_losses_13087

inputs 
gru_cell_12999:`!
gru_cell_13001:	Ќ` 
gru_cell_13003: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂ gru_cell/StatefulPartitionedCallЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2п
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_12999gru_cell_13001gru_cell_13003*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
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
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_12999gru_cell_13001gru_cell_13003*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_13011*
condR
while_cond_13010*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeО
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13001*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulб
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13003*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
І
­
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15039

inputs6
$gru_gru_cell_readvariableop_resource:`9
&gru_gru_cell_readvariableop_1_resource:	Ќ`8
&gru_gru_cell_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identityЂgru/gru_cell/ReadVariableOpЂgru/gru_cell/ReadVariableOp_1Ђgru/gru_cell/ReadVariableOp_2Ђgru/gru_cell/ReadVariableOp_3Ђgru/gru_cell/ReadVariableOp_4Ђgru/gru_cell/ReadVariableOp_5Ђgru/gru_cell/ReadVariableOp_6Ђ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂ	gru/whileЂoutput/BiasAdd/ReadVariableOpЂoutput/Tensordot/ReadVariableOpm
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
!:џџџџџџџџџџџџџџџџџџЌ2
masking/NotEqual
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
masking/Any/reduction_indicesІ
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
masking/Any
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
masking/mul
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
squeeze_dims

џџџџџџџџџ2
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
gru/strided_slice/stack_2њ
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
џџџџџџџџџ2
gru/ExpandDims/dimЄ
gru/ExpandDims
ExpandDimsmasking/Squeeze:output:0gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru/ExpandDims
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/permІ
gru/transpose_1	Transposegru/ExpandDims:output:0gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru/transpose_1
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
gru/TensorArrayV2/element_shapeТ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2Ч
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2;
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
:џџџџџџџџџЌ*
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
gru/gru_cell/ones_like/ConstЙ
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru/gru_cell/ones_like_1/ConstР
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru/gru_cell/mulЁ
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/mul_1Ё
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/gru_cell/mul_2І
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
"gru/gru_cell/strided_slice/stack_2Э
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru/gru_cell/strided_sliceЁ
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/MatMulІ
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
$gru/gru_cell/strided_slice_1/stack_2з
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1Љ
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/MatMul_1І
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
$gru/gru_cell/strided_slice_2/stack_2з
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2Љ
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_3/stack_2К
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_3Џ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_4/stack_2Ј
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_4Е
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_5/stack_2И
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_5Е
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/BiasAdd_2
gru/gru_cell/mul_3Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_3
gru/gru_cell/mul_4Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_4
gru/gru_cell/mul_5Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_5Ѕ
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
$gru/gru_cell/strided_slice_6/stack_2ж
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6Љ
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul_3:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/MatMul_3Ѕ
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
$gru/gru_cell/strided_slice_7/stack_2ж
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7Љ
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_4:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_8/stack_2К
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_8Е
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$gru/gru_cell/strided_slice_9/stack_2Ј
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_9Е
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/BiasAdd_4
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/SigmoidЅ
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add_1
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/Sigmoid_1Ѕ
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
%gru/gru_cell/strided_slice_10/stack_2л
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10Њ
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%gru/gru_cell/strided_slice_11/stack_2Н
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_11Ж
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/BiasAdd_5
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_6
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/Tanh
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru/gru_cell/sub
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/mul_8
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/gru_cell/add_3
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2#
!gru/TensorArrayV2_1/element_shapeШ
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
џџџџџџџџџ2#
!gru/TensorArrayV2_2/element_shapeШ
gru/TensorArrayV2_2TensorListReserve*gru/TensorArrayV2_2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
gru/TensorArrayV2_2Ы
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
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
:џџџџџџџџџ 2
gru/zeros_like
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counterЪ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
gru_while_body_14834* 
condR
gru_while_cond_14833*M
output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *
parallel_iterations 2
	gru/whileН
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4gru/TensorArrayV2Stack/TensorListStack/element_shape
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
gru/strided_slice_3/stack_2В
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
gru/strided_slice_3
gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_2/permО
gru/transpose_2	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
gru/transpose_2n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtimeЋ
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
output/Tensordot/GatherV2/axisє
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
 output/Tensordot/GatherV2_1/axisњ
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
output/Tensordot/Const_1Є
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
output/Tensordot/concat/axisг
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concatЈ
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stackЛ
output/Tensordot/transpose	Transposegru/transpose_2:y:0 output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
output/Tensordot/transposeЛ
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
output/Tensordot/ReshapeК
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
output/Tensordot/concat_1/axisр
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1Е
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
output/TensordotЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЌ
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
output/BiasAddж
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulщ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2:
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
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ы	
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
!:џџџџџџџџџџџџџџџџџџЌ2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Any/reduction_indices
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
mul
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
squeeze_dims

џџџџџџџџџ2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџЌ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
Њ

Ы
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
D: : : : :џџџџџџџџџ :џџџџџџџџџ : :::::: 
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
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ЊВ
Ь
while_body_15690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	Ќ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	Ќ`:
(while_gru_cell_readvariableop_4_resource: `Ђwhile/gru_cell/ReadVariableOpЂwhile/gru_cell/ReadVariableOp_1Ђwhile/gru_cell/ReadVariableOp_2Ђwhile/gru_cell/ReadVariableOp_3Ђwhile/gru_cell/ReadVariableOp_4Ђwhile/gru_cell/ReadVariableOp_5Ђwhile/gru_cell/ReadVariableOp_6У
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
while/gru_cell/ones_like/ConstС
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
 while/gru_cell/ones_like_1/ConstШ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/ones_like_1Ї
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
while/gru_cell/unstackЗ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mulЛ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_1Л
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_2Ў
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
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
$while/gru_cell/strided_slice/stack_2й
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
while/gru_cell/strided_sliceЉ
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMulЎ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stackЁ
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1Ё
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2у
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1Б
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMul_1Ў
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stackЁ
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1Ё
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2у
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2Б
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_3/stack_2Ц
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3З
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_4/stack_2Д
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4Н
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_5/stack_2Ф
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5Н
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_6/stackЁ
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1Ё
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2т
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6Б
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_7/stackЁ
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1Ё
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2т
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7Б
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_8/stack_2Ц
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8Н
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_9/stack_2Д
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9Н
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_4Ї
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%while/gru_cell/strided_slice_10/stackЃ
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1Ѓ
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ч
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10В
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
'while/gru_cell/strided_slice_11/stack_2Щ
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11О
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_5І
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_6Є
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_3м
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
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ш
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2>
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
:џџџџџџџџџ :
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
while_gru_cell_13367_0:	Ќ`(
while_gru_cell_13369_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_13365:`'
while_gru_cell_13367:	Ќ`&
while_gru_cell_13369: `Ђ&while/gru_cell/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
&:џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_132762(
&while/gru_cell/StatefulPartitionedCallѓ
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
while/Identity_2Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4".
while_gru_cell_13365while_gru_cell_13365_0".
while_gru_cell_13367while_gru_cell_13367_0".
while_gru_cell_13369while_gru_cell_13369_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2P
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
т
ђ
.__inference_GRU_classifier_layer_call_fn_14061	
input
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*'
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
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ

_user_specified_nameinput
йФ
ъ
 __inference__wrapped_model_12849	
inputE
3gru_classifier_gru_gru_cell_readvariableop_resource:`H
5gru_classifier_gru_gru_cell_readvariableop_1_resource:	Ќ`G
5gru_classifier_gru_gru_cell_readvariableop_4_resource: `I
7gru_classifier_output_tensordot_readvariableop_resource: C
5gru_classifier_output_biasadd_readvariableop_resource:
identityЂ*GRU_classifier/gru/gru_cell/ReadVariableOpЂ,GRU_classifier/gru/gru_cell/ReadVariableOp_1Ђ,GRU_classifier/gru/gru_cell/ReadVariableOp_2Ђ,GRU_classifier/gru/gru_cell/ReadVariableOp_3Ђ,GRU_classifier/gru/gru_cell/ReadVariableOp_4Ђ,GRU_classifier/gru/gru_cell/ReadVariableOp_5Ђ,GRU_classifier/gru/gru_cell/ReadVariableOp_6ЂGRU_classifier/gru/whileЂ,GRU_classifier/output/BiasAdd/ReadVariableOpЂ.GRU_classifier/output/Tensordot/ReadVariableOp
!GRU_classifier/masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!GRU_classifier/masking/NotEqual/yС
GRU_classifier/masking/NotEqualNotEqualinput*GRU_classifier/masking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2!
GRU_classifier/masking/NotEqualЇ
,GRU_classifier/masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,GRU_classifier/masking/Any/reduction_indicesт
GRU_classifier/masking/AnyAny#GRU_classifier/masking/NotEqual:z:05GRU_classifier/masking/Any/reduction_indices:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
GRU_classifier/masking/AnyЕ
GRU_classifier/masking/CastCast#GRU_classifier/masking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
GRU_classifier/masking/CastЇ
GRU_classifier/masking/mulMulinputGRU_classifier/masking/Cast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
GRU_classifier/masking/mulЫ
GRU_classifier/masking/SqueezeSqueeze#GRU_classifier/masking/Any:output:0*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
squeeze_dims

џџџџџџџџџ2 
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
(GRU_classifier/gru/strided_slice/stack_2д
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
GRU_classifier/gru/zeros/mul/yИ
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
B :ш2!
GRU_classifier/gru/zeros/Less/yГ
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
!GRU_classifier/gru/zeros/packed/1Я
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
GRU_classifier/gru/zeros/ConstС
GRU_classifier/gru/zerosFill(GRU_classifier/gru/zeros/packed:output:0'GRU_classifier/gru/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
GRU_classifier/gru/zeros
!GRU_classifier/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!GRU_classifier/gru/transpose/permе
GRU_classifier/gru/transpose	TransposeGRU_classifier/masking/mul:z:0*GRU_classifier/gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
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
(GRU_classifier/gru/strided_slice_1/stackЂ
*GRU_classifier/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_1/stack_1Ђ
*GRU_classifier/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_1/stack_2р
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
џџџџџџџџџ2#
!GRU_classifier/gru/ExpandDims/dimр
GRU_classifier/gru/ExpandDims
ExpandDims'GRU_classifier/masking/Squeeze:output:0*GRU_classifier/gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
GRU_classifier/gru/ExpandDims
#GRU_classifier/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#GRU_classifier/gru/transpose_1/permт
GRU_classifier/gru/transpose_1	Transpose&GRU_classifier/gru/ExpandDims:output:0,GRU_classifier/gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2 
GRU_classifier/gru/transpose_1Ћ
.GRU_classifier/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.GRU_classifier/gru/TensorArrayV2/element_shapeў
 GRU_classifier/gru/TensorArrayV2TensorListReserve7GRU_classifier/gru/TensorArrayV2/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 GRU_classifier/gru/TensorArrayV2х
HGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2J
HGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
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
(GRU_classifier/gru/strided_slice_2/stackЂ
*GRU_classifier/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_2/stack_1Ђ
*GRU_classifier/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_2/stack_2я
"GRU_classifier/gru/strided_slice_2StridedSlice GRU_classifier/gru/transpose:y:01GRU_classifier/gru/strided_slice_2/stack:output:03GRU_classifier/gru/strided_slice_2/stack_1:output:03GRU_classifier/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_2Е
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
+GRU_classifier/gru/gru_cell/ones_like/Constѕ
%GRU_classifier/gru/gru_cell/ones_likeFill4GRU_classifier/gru/gru_cell/ones_like/Shape:output:04GRU_classifier/gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%GRU_classifier/gru/gru_cell/ones_likeЏ
-GRU_classifier/gru/gru_cell/ones_like_1/ShapeShape!GRU_classifier/gru/zeros:output:0*
T0*
_output_shapes
:2/
-GRU_classifier/gru/gru_cell/ones_like_1/ShapeЃ
-GRU_classifier/gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-GRU_classifier/gru/gru_cell/ones_like_1/Constќ
'GRU_classifier/gru/gru_cell/ones_like_1Fill6GRU_classifier/gru/gru_cell/ones_like_1/Shape:output:06GRU_classifier/gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'GRU_classifier/gru/gru_cell/ones_like_1Ь
*GRU_classifier/gru/gru_cell/ReadVariableOpReadVariableOp3gru_classifier_gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02,
*GRU_classifier/gru/gru_cell/ReadVariableOpО
#GRU_classifier/gru/gru_cell/unstackUnpack2GRU_classifier/gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2%
#GRU_classifier/gru/gru_cell/unstackй
GRU_classifier/gru/gru_cell/mulMul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
GRU_classifier/gru/gru_cell/mulн
!GRU_classifier/gru/gru_cell/mul_1Mul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!GRU_classifier/gru/gru_cell/mul_1н
!GRU_classifier/gru/gru_cell/mul_2Mul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!GRU_classifier/gru/gru_cell/mul_2г
,GRU_classifier/gru/gru_cell/ReadVariableOp_1ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_1Г
/GRU_classifier/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/GRU_classifier/gru/gru_cell/strided_slice/stackЗ
1GRU_classifier/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice/stack_1З
1GRU_classifier/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1GRU_classifier/gru/gru_cell/strided_slice/stack_2Ї
)GRU_classifier/gru/gru_cell/strided_sliceStridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_1:value:08GRU_classifier/gru/gru_cell/strided_slice/stack:output:0:GRU_classifier/gru/gru_cell/strided_slice/stack_1:output:0:GRU_classifier/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2+
)GRU_classifier/gru/gru_cell/strided_sliceн
"GRU_classifier/gru/gru_cell/MatMulMatMul#GRU_classifier/gru/gru_cell/mul:z:02GRU_classifier/gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"GRU_classifier/gru/gru_cell/MatMulг
,GRU_classifier/gru/gru_cell/ReadVariableOp_2ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_2З
1GRU_classifier/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_1/stackЛ
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_1Л
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_2Б
+GRU_classifier/gru/gru_cell/strided_slice_1StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_2:value:0:GRU_classifier/gru/gru_cell/strided_slice_1/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_1/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_1х
$GRU_classifier/gru/gru_cell/MatMul_1MatMul%GRU_classifier/gru/gru_cell/mul_1:z:04GRU_classifier/gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$GRU_classifier/gru/gru_cell/MatMul_1г
,GRU_classifier/gru/gru_cell/ReadVariableOp_3ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_3З
1GRU_classifier/gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   23
1GRU_classifier/gru/gru_cell/strided_slice_2/stackЛ
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_1Л
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_2Б
+GRU_classifier/gru/gru_cell/strided_slice_2StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_3:value:0:GRU_classifier/gru/gru_cell/strided_slice_2/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_2/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_2х
$GRU_classifier/gru/gru_cell/MatMul_2MatMul%GRU_classifier/gru/gru_cell/mul_2:z:04GRU_classifier/gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$GRU_classifier/gru/gru_cell/MatMul_2А
1GRU_classifier/gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_3/stackД
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_1Д
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
+GRU_classifier/gru/gru_cell/strided_slice_3ы
#GRU_classifier/gru/gru_cell/BiasAddBiasAdd,GRU_classifier/gru/gru_cell/MatMul:product:04GRU_classifier/gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#GRU_classifier/gru/gru_cell/BiasAddА
1GRU_classifier/gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_4/stackД
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_1Д
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
+GRU_classifier/gru/gru_cell/strided_slice_4ё
%GRU_classifier/gru/gru_cell/BiasAdd_1BiasAdd.GRU_classifier/gru/gru_cell/MatMul_1:product:04GRU_classifier/gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_1А
1GRU_classifier/gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@23
1GRU_classifier/gru/gru_cell/strided_slice_5/stackД
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_1Д
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
+GRU_classifier/gru/gru_cell/strided_slice_5ё
%GRU_classifier/gru/gru_cell/BiasAdd_2BiasAdd.GRU_classifier/gru/gru_cell/MatMul_2:product:04GRU_classifier/gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_2д
!GRU_classifier/gru/gru_cell/mul_3Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/mul_3д
!GRU_classifier/gru/gru_cell/mul_4Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/mul_4д
!GRU_classifier/gru/gru_cell/mul_5Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/mul_5в
,GRU_classifier/gru/gru_cell/ReadVariableOp_4ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_4З
1GRU_classifier/gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_6/stackЛ
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_1Л
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_2А
+GRU_classifier/gru/gru_cell/strided_slice_6StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_4:value:0:GRU_classifier/gru/gru_cell/strided_slice_6/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_6/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_6х
$GRU_classifier/gru/gru_cell/MatMul_3MatMul%GRU_classifier/gru/gru_cell/mul_3:z:04GRU_classifier/gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$GRU_classifier/gru/gru_cell/MatMul_3в
,GRU_classifier/gru/gru_cell/ReadVariableOp_5ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_5З
1GRU_classifier/gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_7/stackЛ
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_1Л
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_2А
+GRU_classifier/gru/gru_cell/strided_slice_7StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_5:value:0:GRU_classifier/gru/gru_cell/strided_slice_7/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_7/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_7х
$GRU_classifier/gru/gru_cell/MatMul_4MatMul%GRU_classifier/gru/gru_cell/mul_4:z:04GRU_classifier/gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$GRU_classifier/gru/gru_cell/MatMul_4А
1GRU_classifier/gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_8/stackД
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_1Д
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
+GRU_classifier/gru/gru_cell/strided_slice_8ё
%GRU_classifier/gru/gru_cell/BiasAdd_3BiasAdd.GRU_classifier/gru/gru_cell/MatMul_3:product:04GRU_classifier/gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_3А
1GRU_classifier/gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_9/stackД
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_1Д
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
+GRU_classifier/gru/gru_cell/strided_slice_9ё
%GRU_classifier/gru/gru_cell/BiasAdd_4BiasAdd.GRU_classifier/gru/gru_cell/MatMul_4:product:04GRU_classifier/gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_4л
GRU_classifier/gru/gru_cell/addAddV2,GRU_classifier/gru/gru_cell/BiasAdd:output:0.GRU_classifier/gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
GRU_classifier/gru/gru_cell/addЌ
#GRU_classifier/gru/gru_cell/SigmoidSigmoid#GRU_classifier/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#GRU_classifier/gru/gru_cell/Sigmoidс
!GRU_classifier/gru/gru_cell/add_1AddV2.GRU_classifier/gru/gru_cell/BiasAdd_1:output:0.GRU_classifier/gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/add_1В
%GRU_classifier/gru/gru_cell/Sigmoid_1Sigmoid%GRU_classifier/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/gru_cell/Sigmoid_1в
,GRU_classifier/gru/gru_cell/ReadVariableOp_6ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_6Й
2GRU_classifier/gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2GRU_classifier/gru/gru_cell/strided_slice_10/stackН
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_1Н
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_2Е
,GRU_classifier/gru/gru_cell/strided_slice_10StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_6:value:0;GRU_classifier/gru/gru_cell/strided_slice_10/stack:output:0=GRU_classifier/gru/gru_cell/strided_slice_10/stack_1:output:0=GRU_classifier/gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2.
,GRU_classifier/gru/gru_cell/strided_slice_10ц
$GRU_classifier/gru/gru_cell/MatMul_5MatMul%GRU_classifier/gru/gru_cell/mul_5:z:05GRU_classifier/gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$GRU_classifier/gru/gru_cell/MatMul_5В
2GRU_classifier/gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@24
2GRU_classifier/gru/gru_cell/strided_slice_11/stackЖ
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_1Ж
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
,GRU_classifier/gru/gru_cell/strided_slice_11ђ
%GRU_classifier/gru/gru_cell/BiasAdd_5BiasAdd.GRU_classifier/gru/gru_cell/MatMul_5:product:05GRU_classifier/gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_5к
!GRU_classifier/gru/gru_cell/mul_6Mul)GRU_classifier/gru/gru_cell/Sigmoid_1:y:0.GRU_classifier/gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/mul_6и
!GRU_classifier/gru/gru_cell/add_2AddV2.GRU_classifier/gru/gru_cell/BiasAdd_2:output:0%GRU_classifier/gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/add_2Ѕ
 GRU_classifier/gru/gru_cell/TanhTanh%GRU_classifier/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 GRU_classifier/gru/gru_cell/TanhЫ
!GRU_classifier/gru/gru_cell/mul_7Mul'GRU_classifier/gru/gru_cell/Sigmoid:y:0!GRU_classifier/gru/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/mul_7
!GRU_classifier/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!GRU_classifier/gru/gru_cell/sub/xа
GRU_classifier/gru/gru_cell/subSub*GRU_classifier/gru/gru_cell/sub/x:output:0'GRU_classifier/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
GRU_classifier/gru/gru_cell/subЪ
!GRU_classifier/gru/gru_cell/mul_8Mul#GRU_classifier/gru/gru_cell/sub:z:0$GRU_classifier/gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/mul_8Я
!GRU_classifier/gru/gru_cell/add_3AddV2%GRU_classifier/gru/gru_cell/mul_7:z:0%GRU_classifier/gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!GRU_classifier/gru/gru_cell/add_3Е
0GRU_classifier/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
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
GRU_classifier/gru/timeЏ
0GRU_classifier/gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ22
0GRU_classifier/gru/TensorArrayV2_2/element_shape
"GRU_classifier/gru/TensorArrayV2_2TensorListReserve9GRU_classifier/gru/TensorArrayV2_2/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02$
"GRU_classifier/gru/TensorArrayV2_2щ
JGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2L
JGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeЬ
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor"GRU_classifier/gru/transpose_1:y:0SGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02>
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensorЄ
GRU_classifier/gru/zeros_like	ZerosLike%GRU_classifier/gru/gru_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
GRU_classifier/gru/zeros_likeЅ
+GRU_classifier/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+GRU_classifier/gru/while/maximum_iterations
%GRU_classifier/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%GRU_classifier/gru/while/loop_counterК
GRU_classifier/gru/whileWhile.GRU_classifier/gru/while/loop_counter:output:04GRU_classifier/gru/while/maximum_iterations:output:0 GRU_classifier/gru/time:output:0+GRU_classifier/gru/TensorArrayV2_1:handle:0!GRU_classifier/gru/zeros_like:y:0!GRU_classifier/gru/zeros:output:0+GRU_classifier/gru/strided_slice_1:output:0JGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0LGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:03gru_classifier_gru_gru_cell_readvariableop_resource5gru_classifier_gru_gru_cell_readvariableop_1_resource5gru_classifier_gru_gru_cell_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#GRU_classifier_gru_while_body_12656*/
cond'R%
#GRU_classifier_gru_while_cond_12655*M
output_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : *
parallel_iterations 2
GRU_classifier/gru/whileл
CGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2E
CGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shapeН
5GRU_classifier/gru/TensorArrayV2Stack/TensorListStackTensorListStack!GRU_classifier/gru/while:output:3LGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype027
5GRU_classifier/gru/TensorArrayV2Stack/TensorListStackЇ
(GRU_classifier/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2*
(GRU_classifier/gru/strided_slice_3/stackЂ
*GRU_classifier/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*GRU_classifier/gru/strided_slice_3/stack_1Ђ
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
:џџџџџџџџџ *
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_3
#GRU_classifier/gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#GRU_classifier/gru/transpose_2/permњ
GRU_classifier/gru/transpose_2	Transpose>GRU_classifier/gru/TensorArrayV2Stack/TensorListStack:tensor:0,GRU_classifier/gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2 
GRU_classifier/gru/transpose_2
GRU_classifier/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_classifier/gru/runtimeи
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
-GRU_classifier/output/Tensordot/GatherV2/axisП
(GRU_classifier/output/Tensordot/GatherV2GatherV2.GRU_classifier/output/Tensordot/Shape:output:0-GRU_classifier/output/Tensordot/free:output:06GRU_classifier/output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(GRU_classifier/output/Tensordot/GatherV2Є
/GRU_classifier/output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/GRU_classifier/output/Tensordot/GatherV2_1/axisХ
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
%GRU_classifier/output/Tensordot/Constи
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
'GRU_classifier/output/Tensordot/Const_1р
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
&GRU_classifier/output/Tensordot/concatф
%GRU_classifier/output/Tensordot/stackPack-GRU_classifier/output/Tensordot/Prod:output:0/GRU_classifier/output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%GRU_classifier/output/Tensordot/stackї
)GRU_classifier/output/Tensordot/transpose	Transpose"GRU_classifier/gru/transpose_2:y:0/GRU_classifier/output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2+
)GRU_classifier/output/Tensordot/transposeї
'GRU_classifier/output/Tensordot/ReshapeReshape-GRU_classifier/output/Tensordot/transpose:y:0.GRU_classifier/output/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2)
'GRU_classifier/output/Tensordot/Reshapeі
&GRU_classifier/output/Tensordot/MatMulMatMul0GRU_classifier/output/Tensordot/Reshape:output:06GRU_classifier/output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
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
-GRU_classifier/output/Tensordot/concat_1/axisЋ
(GRU_classifier/output/Tensordot/concat_1ConcatV21GRU_classifier/output/Tensordot/GatherV2:output:00GRU_classifier/output/Tensordot/Const_2:output:06GRU_classifier/output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(GRU_classifier/output/Tensordot/concat_1ё
GRU_classifier/output/TensordotReshape0GRU_classifier/output/Tensordot/MatMul:product:01GRU_classifier/output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2!
GRU_classifier/output/TensordotЮ
,GRU_classifier/output/BiasAdd/ReadVariableOpReadVariableOp5gru_classifier_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,GRU_classifier/output/BiasAdd/ReadVariableOpш
GRU_classifier/output/BiasAddBiasAdd(GRU_classifier/output/Tensordot:output:04GRU_classifier/output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
GRU_classifier/output/BiasAddЩ
IdentityIdentity&GRU_classifier/output/BiasAdd:output:0+^GRU_classifier/gru/gru_cell/ReadVariableOp-^GRU_classifier/gru/gru_cell/ReadVariableOp_1-^GRU_classifier/gru/gru_cell/ReadVariableOp_2-^GRU_classifier/gru/gru_cell/ReadVariableOp_3-^GRU_classifier/gru/gru_cell/ReadVariableOp_4-^GRU_classifier/gru/gru_cell/ReadVariableOp_5-^GRU_classifier/gru/gru_cell/ReadVariableOp_6^GRU_classifier/gru/while-^GRU_classifier/output/BiasAdd/ReadVariableOp/^GRU_classifier/output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2X
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
!:џџџџџџџџџџџџџџџџџџЌ

_user_specified_nameinput
Ю
Т
>__inference_gru_layer_call_and_return_conditional_losses_13991

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	Ќ`4
"gru_cell_readvariableop_4_resource: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂgru_cell/ReadVariableOpЂgru_cell/ReadVariableOp_1Ђgru_cell/ReadVariableOp_2Ђgru_cell/ReadVariableOp_3Ђgru_cell/ReadVariableOp_4Ђgru_cell/ReadVariableOp_5Ђgru_cell/ReadVariableOp_6ЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
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
gru_cell/ones_like/ConstЉ
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru_cell/ones_like_1/ConstА
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
gru_cell/strided_slice/stack_2Е
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_1/stack_2П
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_2/stack_2П
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_3/stack_2Ђ
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
:џџџџџџџџџ 2
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
gru_cell/strided_slice_4Ѕ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_5Ѕ
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_6/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_7/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_8/stack_2Ђ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8Ѕ
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_9Ѕ
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_10/stack_2У
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
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_11/stack_2Ѕ
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11І
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
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
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_13827*
condR
while_cond_13826*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeв
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulх
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulД
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
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
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
'
Е
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14536

inputs
	gru_14511:`
	gru_14513:	Ќ`
	gru_14515: `
output_14518: 
output_14520:
identityЂgru/StatefulPartitionedCallЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂoutput/StatefulPartitionedCallу
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCallБ
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_14511	gru_14513	gru_14515*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_144752
gru/StatefulPartitionedCallЗ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14518output_14520*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCallЙ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14513*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulЬ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14515*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulС
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
х
ѓ
.__inference_GRU_classifier_layer_call_fn_14687

inputs
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*'
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
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs

Д
#__inference_gru_layer_call_fn_15526
inputs_0
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
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
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
Ў
ч
#__inference_signature_wrapper_14657	
input
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*'
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
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ

_user_specified_nameinput
Ф
Й
__inference_loss_fn_0_17297Q
>gru_gru_cell_kernel_regularizer_square_readvariableop_resource:	Ќ`
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpю
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>gru_gru_cell_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulЂ
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
їљ
Ь
while_body_16719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	Ќ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	Ќ`:
(while_gru_cell_readvariableop_4_resource: `Ђwhile/gru_cell/ReadVariableOpЂwhile/gru_cell/ReadVariableOp_1Ђwhile/gru_cell/ReadVariableOp_2Ђwhile/gru_cell/ReadVariableOp_3Ђwhile/gru_cell/ReadVariableOp_4Ђwhile/gru_cell/ReadVariableOp_5Ђwhile/gru_cell/ReadVariableOp_6У
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
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
while/gru_cell/ones_like/ConstС
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/ones_like
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/ConstМ
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/Mul
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeў
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ьЯ25
3while/gru_cell/dropout/random_uniform/RandomUniform
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2'
%while/gru_cell/dropout/GreaterEqual/yћ
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#while/gru_cell/dropout/GreaterEqual­
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/CastЗ
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout/Mul_1
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/ConstТ
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_1/Mul
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2јй27
5while/gru_cell/dropout_1/random_uniform/RandomUniform
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_1/GreaterEqual/y
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%while/gru_cell/dropout_1/GreaterEqualГ
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_1/CastП
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
while/gru_cell/dropout_1/Mul_1
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/ConstТ
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_2/Mul
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2Ђ27
5while/gru_cell/dropout_2/random_uniform/RandomUniform
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_2/GreaterEqual/y
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%while/gru_cell/dropout_2/GreaterEqualГ
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/dropout_2/CastП
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
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
 while/gru_cell/ones_like_1/ConstШ
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/ones_like_1
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_3/ConstУ
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_3/Mul
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2н27
5while/gru_cell/dropout_3/random_uniform/RandomUniform
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_3/GreaterEqual/y
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_3/GreaterEqualВ
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_3/CastО
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_3/Mul_1
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_4/ConstУ
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_4/Mul
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2єГ27
5while/gru_cell/dropout_4/random_uniform/RandomUniform
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_4/GreaterEqual/y
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_4/GreaterEqualВ
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_4/CastО
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_4/Mul_1
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_5/ConstУ
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_5/Mul
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2Щљ27
5while/gru_cell/dropout_5/random_uniform/RandomUniform
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2)
'while/gru_cell/dropout_5/GreaterEqual/y
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%while/gru_cell/dropout_5/GreaterEqualВ
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/dropout_5/CastО
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/gru_cell/dropout_5/Mul_1Ї
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
while/gru_cell/unstackЖ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mulМ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_1М
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/gru_cell/mul_2Ў
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
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
$while/gru_cell/strided_slice/stack_2й
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
while/gru_cell/strided_sliceЉ
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMulЎ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_2
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stackЁ
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1Ё
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2у
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1Б
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/MatMul_1Ў
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02!
while/gru_cell/ReadVariableOp_3
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stackЁ
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1Ё
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2у
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2Б
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_3/stack_2Ц
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3З
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_4/stack_2Д
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4Н
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_5/stack_2Ф
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5Н
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_2
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_3
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_4
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_6/stackЁ
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1Ё
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2т
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6Б
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
$while/gru_cell/strided_slice_7/stackЁ
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1Ё
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2т
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7Б
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_8/stack_2Ц
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8Н
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
&while/gru_cell/strided_slice_9/stack_2Д
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9Н
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_4Ї
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Sigmoid­
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
%while/gru_cell/strided_slice_10/stackЃ
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1Ѓ
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2ч
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10В
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
'while/gru_cell/strided_slice_11/stack_2Щ
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11О
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/BiasAdd_5І
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_6Є
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/Tanh
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
while/gru_cell/sub
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/mul_8
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/gru_cell/add_3м
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
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3ш
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ : : : : : 2>
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
х
ѓ
.__inference_GRU_classifier_layer_call_fn_14672

inputs
unknown:`
	unknown_0:	Ќ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*'
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
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
ѕ
Ѕ
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
-: : : : :џџџџџџџџџ : ::::: 
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
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Е

C__inference_gru_cell_layer_call_and_return_conditional_losses_12998

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	Ќ`+
readvariableop_4_resource: `
identity

identity_1ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4ЂReadVariableOp_5ЂReadVariableOp_6Ђ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
:џџџџџџџџџЌ2
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
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
strided_slice/stack_2џ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

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
strided_slice_3/stack_2ь
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
:џџџџџџџџџ 2	
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
strided_slice_4/stack_2к
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
strided_slice_5/stack_2ъ
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
:џџџџџџџџџ 2
	BiasAdd_2e
mul_3Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
:џџџџџџџџџ 2

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
strided_slice_8/stack_2ь
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
:џџџџџџџџџ 2
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
strided_slice_9/stack_2к
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
strided_slice_11/stack_2я
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
:џџџџџџџџџ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Щ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulм
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulк
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџ : : : 2 
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
:џџџџџџџџџЌ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
'
Д
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14622	
input
	gru_14597:`
	gru_14599:	Ќ`
	gru_14601: `
output_14604: 
output_14606:
identityЂgru/StatefulPartitionedCallЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂoutput/StatefulPartitionedCallт
masking/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8 *K
fFRD
B__inference_masking_layer_call_and_return_conditional_losses_136952
masking/PartitionedCallБ
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_14597	gru_14599	gru_14601*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_144752
gru/StatefulPartitionedCallЗ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_14604output_14606*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_140292 
output/StatefulPartitionedCallЙ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14599*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulЬ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_14601*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulС
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџџџџџџџџџџЌ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ

_user_specified_nameinput
ЃЛ

C__inference_gru_cell_layer_call_and_return_conditional_losses_13276

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	Ќ`+
readvariableop_4_resource: `
identity

identity_1ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4ЂReadVariableOp_5ЂReadVariableOp_6Ђ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
:џџџџџџџџџЌ2
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
:џџџџџџџџџЌ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeб
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2ѕ­Ђ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
:џџџџџџџџџЌ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeж
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2Љx2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_1/GreaterEqual/yЧ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
:џџџџџџџџџЌ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeз
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*

seedJ*
seed2И2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_2/GreaterEqual/yЧ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeж
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2бТ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeе
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2їЗ@2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeж
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*

seedJ*
seed2Кщ2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
strided_slice/stack_2џ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

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
strided_slice_3/stack_2ь
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
:џџџџџџџџџ 2	
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
strided_slice_4/stack_2к
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
strided_slice_5/stack_2ъ
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
:џџџџџџџџџ 2
	BiasAdd_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
:џџџџџџџџџ 2

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
strided_slice_8/stack_2ь
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
:џџџџџџџџџ 2
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
strided_slice_9/stack_2к
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
strided_slice_11/stack_2я
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
:џџџџџџџџџ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Щ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulм
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulк
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџ : : : 2 
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
:џџџџџџџџџЌ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
У

C__inference_gru_cell_layer_call_and_return_conditional_losses_17124

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	Ќ`+
readvariableop_4_resource: `
identity

identity_1ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4ЂReadVariableOp_5ЂReadVariableOp_6Ђ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
:џџџџџџџџџЌ2
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
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
strided_slice/stack_2џ
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
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
:	Ќ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

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
strided_slice_3/stack_2ь
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
:џџџџџџџџџ 2	
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
strided_slice_4/stack_2к
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
strided_slice_5/stack_2ъ
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
:џџџџџџџџџ 2
	BiasAdd_2g
mul_3Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
:џџџџџџџџџ 2

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
strided_slice_8/stack_2ь
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
:џџџџџџџџџ 2
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
strided_slice_9/stack_2к
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2

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
strided_slice_11/stack_2я
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
:џџџџџџџџџ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Щ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulм
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulк
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityо

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџЌ:џџџџџџџџџ : : : 2 
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
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0
эд
г
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
.gru_while_gru_cell_readvariableop_1_resource_0:	Ќ`@
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
,gru_while_gru_cell_readvariableop_1_resource:	Ќ`>
,gru_while_gru_cell_readvariableop_4_resource: `Ђ!gru/while/gru_cell/ReadVariableOpЂ#gru/while/gru_cell/ReadVariableOp_1Ђ#gru/while/gru_cell/ReadVariableOp_2Ђ#gru/while/gru_cell/ReadVariableOp_3Ђ#gru/while/gru_cell/ReadVariableOp_4Ђ#gru/while/gru_cell/ReadVariableOp_5Ђ#gru/while/gru_cell/ReadVariableOp_6Ы
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeь
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItemЯ
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeѕ
/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0gru_while_placeholderFgru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
21
/gru/while/TensorArrayV2Read_1/TensorListGetItemЌ
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
"gru/while/gru_cell/ones_like/Constб
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
$gru/while/gru_cell/ones_like_1/Constи
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
gru/while/gru_cell/ones_like_1Г
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02#
!gru/while/gru_cell/ReadVariableOpЃ
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell/unstackЧ
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/mulЫ
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/mul_1Ы
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru/while/gru_cell/mul_2К
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1Ё
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stackЅ
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice/stack_1Ѕ
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2ё
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_sliceЙ
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMulК
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2Ѕ
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_1/stackЉ
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_1/stack_1Љ
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2ћ
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1С
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_1К
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	Ќ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3Ѕ
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(gru/while/gru_cell/strided_slice_2/stackЉ
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1Љ
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2ћ
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2С
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_2
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stackЂ
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_3/stack_1Ђ
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2о
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_3Ч
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_4/stackЂ
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_4/stack_1Ђ
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2Ь
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_4Э
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_1
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2*
(gru/while/gru_cell/strided_slice_5/stackЂ
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1Ђ
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2м
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2$
"gru/while/gru_cell/strided_slice_5Э
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_2Џ
gru/while/gru_cell/mul_3Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_3Џ
gru/while/gru_cell/mul_4Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_4Џ
gru/while/gru_cell/mul_5Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_5Й
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4Ѕ
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stackЉ
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_6/stack_1Љ
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2њ
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6С
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_3Й
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5Ѕ
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_7/stackЉ
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*gru/while/gru_cell/strided_slice_7/stack_1Љ
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2њ
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7С
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_4
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stackЂ
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_8/stack_1Ђ
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2о
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_8Э
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_3
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_9/stackЂ
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_9/stack_1Ђ
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2Ь
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_9Э
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_4З
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/SigmoidН
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add_1
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/Sigmoid_1Й
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6Ї
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)gru/while/gru_cell/strided_slice_10/stackЋ
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1Ћ
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2џ
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10Т
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/MatMul_5 
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2+
)gru/while/gru_cell/strided_slice_11/stackЄ
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1Є
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2с
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2%
#gru/while/gru_cell/strided_slice_11Ю
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/BiasAdd_5Ж
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_6Д
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add_2
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/TanhІ
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell/sub/xЌ
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/subІ
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/mul_8Ћ
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/gru_cell/add_3
gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile/multiplesЕ
gru/while/TileTile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0!gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
gru/while/TileЖ
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/SelectV2
gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile_1/multiplesЛ
gru/while/Tile_1Tile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0#gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
gru/while/Tile_1М
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/SelectV2_1я
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
gru/while/add_1ђ
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1є
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2Ё
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/SelectV2:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru/while/Identity_4
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_while_identity_5gru/while/Identity_5:output:0"Р
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0"И
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : : 2F
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
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ю
Т
>__inference_gru_layer_call_and_return_conditional_losses_16540

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	Ќ`4
"gru_cell_readvariableop_4_resource: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂgru_cell/ReadVariableOpЂgru_cell/ReadVariableOp_1Ђgru_cell/ReadVariableOp_2Ђgru_cell/ReadVariableOp_3Ђgru_cell/ReadVariableOp_4Ђgru_cell/ReadVariableOp_5Ђgru_cell/ReadVariableOp_6ЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
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
gru_cell/ones_like/ConstЉ
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
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
gru_cell/ones_like_1/ConstА
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџЌ2
gru_cell/mul
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_1
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
gru_cell/mul_2
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
gru_cell/strided_slice/stack_2Е
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_1/stack_2П
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_1
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/MatMul_1
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
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
 gru_cell/strided_slice_2/stack_2П
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	Ќ *

begin_mask*
end_mask2
gru_cell/strided_slice_2
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_3/stack_2Ђ
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
:џџџџџџџџџ 2
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
gru_cell/strided_slice_4Ѕ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_5Ѕ
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_2
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_3
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_4
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_6/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_7/stack_2О
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
:џџџџџџџџџ 2
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
 gru_cell/strided_slice_8/stack_2Ђ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8Ѕ
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
gru_cell/strided_slice_9Ѕ
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_4
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_10/stack_2У
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
:џџџџџџџџџ 2
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
!gru_cell/strided_slice_11/stack_2Ѕ
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11І
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/BiasAdd_5
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_6
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
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
:џџџџџџџџџ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/mul_8
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
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
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_16376*
condR
while_cond_16375*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeв
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulх
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulД
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
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
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
йS
щ
>__inference_gru_layer_call_and_return_conditional_losses_13419

inputs 
gru_cell_13331:`!
gru_cell_13333:	Ќ` 
gru_cell_13335: `
identityЂ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpЂ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЂ gru_cell/StatefulPartitionedCallЂwhileD
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
strided_slice/stack_2т
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
B :ш2
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
:џџџџџџџџџ 2
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
!:џџџџџџџџџџџџџџџџџџЌ2
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
strided_slice_1/stack_2ю
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
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
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
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2п
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_13331gru_cell_13333gru_cell_13335*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ *%
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
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
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
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterй
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_13331gru_cell_13333gru_cell_13335*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_13343*
condR
while_cond_13342*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
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
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeО
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13333*
_output_shapes
:	Ќ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpУ
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ќ`2(
&gru/gru_cell/kernel/Regularizer/Square
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/ConstЮ
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
з#<2'
%gru/gru_cell/kernel/Regularizer/mul/xа
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulб
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_13335*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpр
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/SquareГ
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constі
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumЇ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xј
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mul
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџЌ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs"ЬL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Р
serving_defaultЌ
E
input<
serving_default_input:0џџџџџџџџџџџџџџџџџџЌG
output=
StatefulPartitionedCall:0џџџџџџџџџџџџџџџџџџtensorflow/serving/predict:ћШ
ј0
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
*X&call_and_return_all_conditional_losses"Х.
_tf_keras_networkЉ.{"name": "GRU_classifier", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "GRU_classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "name": "masking", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["masking", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["gru", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 300]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "GRU_classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "name": "masking", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["masking", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["gru", 0, 0, {}]]], "shared_object_id": 10}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 13}, "metrics": [[{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 14}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00039999998989515007, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ѕ"ђ
_tf_keras_input_layerв{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}

regularization_losses
trainable_variables
	variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"ј
_tf_keras_layerо{"name": "masking", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "stateful": false, "must_restore_from_config": false, "class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}
ђ
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"Щ
_tf_keras_rnn_layerЋ{"name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "inbound_nodes": [[["masking", 0, 0, {}]]], "shared_object_id": 7, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}}
ћ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["gru", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 32]}}
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
Ъ
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
*a&call_and_return_all_conditional_losses"л
_tf_keras_layerС{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 6}
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
Й
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
&:$	Ќ`2gru/gru_cell/kernel
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
д
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 17}
Ї
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"р
_tf_keras_metricХ{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 14}
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
+:)	Ќ`2Adam/gru/gru_cell/kernel/m
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/m
(:&`2Adam/gru/gru_cell/bias/m
$:" 2Adam/output/kernel/v
:2Adam/output/bias/v
+:)	Ќ`2Adam/gru/gru_cell/kernel/v
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/v
(:&`2Adam/gru/gru_cell/bias/v
ъ2ч
 __inference__wrapped_model_12849Т
В
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
annotationsЊ *2Ђ/
-*
inputџџџџџџџџџџџџџџџџџџЌ
2
.__inference_GRU_classifier_layer_call_fn_14061
.__inference_GRU_classifier_layer_call_fn_14672
.__inference_GRU_classifier_layer_call_fn_14687
.__inference_GRU_classifier_layer_call_fn_14564Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15039
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15487
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14593
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14622Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_masking_layer_call_fn_15492Ђ
В
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
annotationsЊ *
 
ь2щ
B__inference_masking_layer_call_and_return_conditional_losses_15503Ђ
В
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
annotationsЊ *
 
я2ь
#__inference_gru_layer_call_fn_15526
#__inference_gru_layer_call_fn_15537
#__inference_gru_layer_call_fn_15548
#__inference_gru_layer_call_fn_15559е
ЬВШ
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
kwonlydefaultsЊ 
annotationsЊ *
 
л2и
>__inference_gru_layer_call_and_return_conditional_losses_15854
>__inference_gru_layer_call_and_return_conditional_losses_16245
>__inference_gru_layer_call_and_return_conditional_losses_16540
>__inference_gru_layer_call_and_return_conditional_losses_16931е
ЬВШ
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
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
&__inference_output_layer_call_fn_16940Ђ
В
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
annotationsЊ *
 
ы2ш
A__inference_output_layer_call_and_return_conditional_losses_16970Ђ
В
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
annotationsЊ *
 
ШBХ
#__inference_signature_wrapper_14657input"
В
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
annotationsЊ *
 
2
(__inference_gru_cell_layer_call_fn_16996
(__inference_gru_cell_layer_call_fn_17010О
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
C__inference_gru_cell_layer_call_and_return_conditional_losses_17124
C__inference_gru_cell_layer_call_and_return_conditional_losses_17286О
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
В2Џ
__inference_loss_fn_0_17297
В
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
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_1_17308
В
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
annotationsЊ *Ђ Я
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14593" !DЂA
:Ђ7
-*
inputџџџџџџџџџџџџџџџџџџЌ
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Я
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_14622" !DЂA
:Ђ7
-*
inputџџџџџџџџџџџџџџџџџџЌ
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 а
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15039" !EЂB
;Ђ8
.+
inputsџџџџџџџџџџџџџџџџџџЌ
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 а
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_15487" !EЂB
;Ђ8
.+
inputsџџџџџџџџџџџџџџџџџџЌ
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 І
.__inference_GRU_classifier_layer_call_fn_14061t" !DЂA
:Ђ7
-*
inputџџџџџџџџџџџџџџџџџџЌ
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџІ
.__inference_GRU_classifier_layer_call_fn_14564t" !DЂA
:Ђ7
-*
inputџџџџџџџџџџџџџџџџџџЌ
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЇ
.__inference_GRU_classifier_layer_call_fn_14672u" !EЂB
;Ђ8
.+
inputsџџџџџџџџџџџџџџџџџџЌ
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџЇ
.__inference_GRU_classifier_layer_call_fn_14687u" !EЂB
;Ђ8
.+
inputsџџџџџџџџџџџџџџџџџџЌ
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЈ
 __inference__wrapped_model_12849" !<Ђ9
2Ђ/
-*
inputџџџџџџџџџџџџџџџџџџЌ
Њ "<Њ9
7
output-*
outputџџџџџџџџџџџџџџџџџџ
C__inference_gru_cell_layer_call_and_return_conditional_losses_17124И" !]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџ 
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ 
$!

0/1/0џџџџџџџџџ 
 
C__inference_gru_cell_layer_call_and_return_conditional_losses_17286И" !]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџ 
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ 
$!

0/1/0џџџџџџџџџ 
 з
(__inference_gru_cell_layer_call_fn_16996Њ" !]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџ 
p 
Њ "DЂA

0џџџџџџџџџ 
"

1/0џџџџџџџџџ з
(__inference_gru_cell_layer_call_fn_17010Њ" !]ЂZ
SЂP
!
inputsџџџџџџџџџЌ
'Ђ$
"
states/0џџџџџџџџџ 
p
Њ "DЂA

0џџџџџџџџџ 
"

1/0џџџџџџџџџ Ю
>__inference_gru_layer_call_and_return_conditional_losses_15854" !PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ 
 Ю
>__inference_gru_layer_call_and_return_conditional_losses_16245" !PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ 
 Ч
>__inference_gru_layer_call_and_return_conditional_losses_16540" !IЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ 
 Ч
>__inference_gru_layer_call_and_return_conditional_losses_16931" !IЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ 
 Ѕ
#__inference_gru_layer_call_fn_15526~" !PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџ Ѕ
#__inference_gru_layer_call_fn_15537~" !PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ 
#__inference_gru_layer_call_fn_15548w" !IЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџ 
#__inference_gru_layer_call_fn_15559w" !IЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ :
__inference_loss_fn_0_17297 Ђ

Ђ 
Њ " :
__inference_loss_fn_1_17308!Ђ

Ђ 
Њ " К
B__inference_masking_layer_call_and_return_conditional_losses_15503t=Ђ:
3Ђ0
.+
inputsџџџџџџџџџџџџџџџџџџЌ
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџЌ
 
'__inference_masking_layer_call_fn_15492g=Ђ:
3Ђ0
.+
inputsџџџџџџџџџџџџџџџџџџЌ
Њ "&#џџџџџџџџџџџџџџџџџџЌЛ
A__inference_output_layer_call_and_return_conditional_losses_16970v<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 
&__inference_output_layer_call_fn_16940i<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ 
Њ "%"џџџџџџџџџџџџџџџџџџД
#__inference_signature_wrapper_14657" !EЂB
Ђ 
;Њ8
6
input-*
inputџџџџџџџџџџџџџџџџџџЌ"<Њ9
7
output-*
outputџџџџџџџџџџџџџџџџџџ
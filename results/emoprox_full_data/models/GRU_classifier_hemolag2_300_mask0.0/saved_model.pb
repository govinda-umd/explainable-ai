аУ1
Ѕ С 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
dtypetypeИ
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
≠
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
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
Н
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
Т
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
Б
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718©І/
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
Г
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ`*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	ђ`*
dtype0
Ц
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*.
shared_namegru/gru_cell/recurrent_kernel
П
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
Д
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
С
Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ`*+
shared_nameAdam/gru/gru_cell/kernel/m
К
.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes
:	ђ`*
dtype0
§
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m
Э
8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0
М
Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/m
Е
,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
_output_shapes

:`*
dtype0
Д
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
С
Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ`*+
shared_nameAdam/gru/gru_cell/kernel/v
К
.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes
:	ђ`*
dtype0
§
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v
Э
8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0
М
Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/v
Е
,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes

:`*
dtype0

NoOpNoOp
і$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*п#
valueе#Bв# Bџ#
ў
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Ъ
iter

beta_1

beta_2
	decay
learning_ratemLmM mN!mO"mPvQvR vS!vT"vU
#
 0
!1
"2
3
4
 
#
 0
!1
"2
3
4
≠
#non_trainable_variables

$layers
%layer_regularization_losses
	variables
&layer_metrics
'metrics
regularization_losses
trainable_variables
 
 
 
 
≠
(non_trainable_variables

)layers
*layer_regularization_losses
	variables
+layer_metrics
trainable_variables
,metrics
regularization_losses
~

 kernel
!recurrent_kernel
"bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
 

 0
!1
"2

 0
!1
"2
 
є

1layers
2non_trainable_variables
3layer_regularization_losses
	variables
4layer_metrics
trainable_variables
5metrics
regularization_losses

6states
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
7non_trainable_variables

8layers
9layer_regularization_losses
	variables
:layer_metrics
trainable_variables
;metrics
regularization_losses
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
OM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 
 

<0
=1
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
 
≠
>non_trainable_variables

?layers
@layer_regularization_losses
-	variables
Alayer_metrics
.trainable_variables
Bmetrics
/regularization_losses
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
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ф
serving_default_inputPlaceholder*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
dtype0**
shape!:€€€€€€€€€€€€€€€€€€ђ
Ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kerneloutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8В *,
f'R%
#__inference_signature_wrapper_10757
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
л	
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
GPU2 *1J 8В *'
f"R 
__inference__traced_save_13503
Ж
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
GPU2 *1J 8В **
f%R#
!__inference__traced_restore_13585“њ.
е
у
.__inference_GRU_classifier_layer_call_fn_11587

inputs
unknown:`
	unknown_0:	ђ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8В *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_106362
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
÷ƒ
й
__inference__wrapped_model_8949	
inputE
3gru_classifier_gru_gru_cell_readvariableop_resource:`H
5gru_classifier_gru_gru_cell_readvariableop_1_resource:	ђ`G
5gru_classifier_gru_gru_cell_readvariableop_4_resource: `I
7gru_classifier_output_tensordot_readvariableop_resource: C
5gru_classifier_output_biasadd_readvariableop_resource:
identityИҐ*GRU_classifier/gru/gru_cell/ReadVariableOpҐ,GRU_classifier/gru/gru_cell/ReadVariableOp_1Ґ,GRU_classifier/gru/gru_cell/ReadVariableOp_2Ґ,GRU_classifier/gru/gru_cell/ReadVariableOp_3Ґ,GRU_classifier/gru/gru_cell/ReadVariableOp_4Ґ,GRU_classifier/gru/gru_cell/ReadVariableOp_5Ґ,GRU_classifier/gru/gru_cell/ReadVariableOp_6ҐGRU_classifier/gru/whileҐ,GRU_classifier/output/BiasAdd/ReadVariableOpҐ.GRU_classifier/output/Tensordot/ReadVariableOpЛ
!GRU_classifier/masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!GRU_classifier/masking/NotEqual/yЅ
GRU_classifier/masking/NotEqualNotEqualinput*GRU_classifier/masking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2!
GRU_classifier/masking/NotEqualІ
,GRU_classifier/masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,GRU_classifier/masking/Any/reduction_indicesв
GRU_classifier/masking/AnyAny#GRU_classifier/masking/NotEqual:z:05GRU_classifier/masking/Any/reduction_indices:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
	keep_dims(2
GRU_classifier/masking/Anyµ
GRU_classifier/masking/CastCast#GRU_classifier/masking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
GRU_classifier/masking/CastІ
GRU_classifier/masking/mulMulinputGRU_classifier/masking/Cast:y:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
GRU_classifier/masking/mulЋ
GRU_classifier/masking/SqueezeSqueeze#GRU_classifier/masking/Any:output:0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
squeeze_dims

€€€€€€€€€2 
GRU_classifier/masking/SqueezeВ
GRU_classifier/gru/ShapeShapeGRU_classifier/masking/mul:z:0*
T0*
_output_shapes
:2
GRU_classifier/gru/ShapeЪ
&GRU_classifier/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&GRU_classifier/gru/strided_slice/stackЮ
(GRU_classifier/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(GRU_classifier/gru/strided_slice/stack_1Ю
(GRU_classifier/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(GRU_classifier/gru/strided_slice/stack_2‘
 GRU_classifier/gru/strided_sliceStridedSlice!GRU_classifier/gru/Shape:output:0/GRU_classifier/gru/strided_slice/stack:output:01GRU_classifier/gru/strided_slice/stack_1:output:01GRU_classifier/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 GRU_classifier/gru/strided_sliceВ
GRU_classifier/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2 
GRU_classifier/gru/zeros/mul/yЄ
GRU_classifier/gru/zeros/mulMul)GRU_classifier/gru/strided_slice:output:0'GRU_classifier/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_classifier/gru/zeros/mulЕ
GRU_classifier/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2!
GRU_classifier/gru/zeros/Less/y≥
GRU_classifier/gru/zeros/LessLess GRU_classifier/gru/zeros/mul:z:0(GRU_classifier/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_classifier/gru/zeros/LessИ
!GRU_classifier/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!GRU_classifier/gru/zeros/packed/1ѕ
GRU_classifier/gru/zeros/packedPack)GRU_classifier/gru/strided_slice:output:0*GRU_classifier/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
GRU_classifier/gru/zeros/packedЕ
GRU_classifier/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
GRU_classifier/gru/zeros/ConstЅ
GRU_classifier/gru/zerosFill(GRU_classifier/gru/zeros/packed:output:0'GRU_classifier/gru/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
GRU_classifier/gru/zerosЫ
!GRU_classifier/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!GRU_classifier/gru/transpose/perm’
GRU_classifier/gru/transpose	TransposeGRU_classifier/masking/mul:z:0*GRU_classifier/gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
GRU_classifier/gru/transposeИ
GRU_classifier/gru/Shape_1Shape GRU_classifier/gru/transpose:y:0*
T0*
_output_shapes
:2
GRU_classifier/gru/Shape_1Ю
(GRU_classifier/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(GRU_classifier/gru/strided_slice_1/stackҐ
*GRU_classifier/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_1/stack_1Ґ
*GRU_classifier/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_1/stack_2а
"GRU_classifier/gru/strided_slice_1StridedSlice#GRU_classifier/gru/Shape_1:output:01GRU_classifier/gru/strided_slice_1/stack:output:03GRU_classifier/gru/strided_slice_1/stack_1:output:03GRU_classifier/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_1С
!GRU_classifier/gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2#
!GRU_classifier/gru/ExpandDims/dimа
GRU_classifier/gru/ExpandDims
ExpandDims'GRU_classifier/masking/Squeeze:output:0*GRU_classifier/gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
GRU_classifier/gru/ExpandDimsЯ
#GRU_classifier/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#GRU_classifier/gru/transpose_1/permв
GRU_classifier/gru/transpose_1	Transpose&GRU_classifier/gru/ExpandDims:output:0,GRU_classifier/gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2 
GRU_classifier/gru/transpose_1Ђ
.GRU_classifier/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€20
.GRU_classifier/gru/TensorArrayV2/element_shapeю
 GRU_classifier/gru/TensorArrayV2TensorListReserve7GRU_classifier/gru/TensorArrayV2/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 GRU_classifier/gru/TensorArrayV2е
HGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  2J
HGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeƒ
:GRU_classifier/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor GRU_classifier/gru/transpose:y:0QGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:GRU_classifier/gru/TensorArrayUnstack/TensorListFromTensorЮ
(GRU_classifier/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(GRU_classifier/gru/strided_slice_2/stackҐ
*GRU_classifier/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_2/stack_1Ґ
*GRU_classifier/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_2/stack_2п
"GRU_classifier/gru/strided_slice_2StridedSlice GRU_classifier/gru/transpose:y:01GRU_classifier/gru/strided_slice_2/stack:output:03GRU_classifier/gru/strided_slice_2/stack_1:output:03GRU_classifier/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_2µ
+GRU_classifier/gru/gru_cell/ones_like/ShapeShape+GRU_classifier/gru/strided_slice_2:output:0*
T0*
_output_shapes
:2-
+GRU_classifier/gru/gru_cell/ones_like/ShapeЯ
+GRU_classifier/gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+GRU_classifier/gru/gru_cell/ones_like/Constх
%GRU_classifier/gru/gru_cell/ones_likeFill4GRU_classifier/gru/gru_cell/ones_like/Shape:output:04GRU_classifier/gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%GRU_classifier/gru/gru_cell/ones_likeѓ
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
 *  А?2/
-GRU_classifier/gru/gru_cell/ones_like_1/Constь
'GRU_classifier/gru/gru_cell/ones_like_1Fill6GRU_classifier/gru/gru_cell/ones_like_1/Shape:output:06GRU_classifier/gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/gru_cell/ones_like_1ћ
*GRU_classifier/gru/gru_cell/ReadVariableOpReadVariableOp3gru_classifier_gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02,
*GRU_classifier/gru/gru_cell/ReadVariableOpЊ
#GRU_classifier/gru/gru_cell/unstackUnpack2GRU_classifier/gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2%
#GRU_classifier/gru/gru_cell/unstackў
GRU_classifier/gru/gru_cell/mulMul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
GRU_classifier/gru/gru_cell/mulЁ
!GRU_classifier/gru/gru_cell/mul_1Mul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2#
!GRU_classifier/gru/gru_cell/mul_1Ё
!GRU_classifier/gru/gru_cell/mul_2Mul+GRU_classifier/gru/strided_slice_2:output:0.GRU_classifier/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2#
!GRU_classifier/gru/gru_cell/mul_2”
,GRU_classifier/gru/gru_cell/ReadVariableOp_1ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_1≥
/GRU_classifier/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/GRU_classifier/gru/gru_cell/strided_slice/stackЈ
1GRU_classifier/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice/stack_1Ј
1GRU_classifier/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1GRU_classifier/gru/gru_cell/strided_slice/stack_2І
)GRU_classifier/gru/gru_cell/strided_sliceStridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_1:value:08GRU_classifier/gru/gru_cell/strided_slice/stack:output:0:GRU_classifier/gru/gru_cell/strided_slice/stack_1:output:0:GRU_classifier/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2+
)GRU_classifier/gru/gru_cell/strided_sliceЁ
"GRU_classifier/gru/gru_cell/MatMulMatMul#GRU_classifier/gru/gru_cell/mul:z:02GRU_classifier/gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"GRU_classifier/gru/gru_cell/MatMul”
,GRU_classifier/gru/gru_cell/ReadVariableOp_2ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_2Ј
1GRU_classifier/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_1/stackї
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell/strided_slice_1/stack_1ї
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
:	ђ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_1е
$GRU_classifier/gru/gru_cell/MatMul_1MatMul%GRU_classifier/gru/gru_cell/mul_1:z:04GRU_classifier/gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$GRU_classifier/gru/gru_cell/MatMul_1”
,GRU_classifier/gru/gru_cell/ReadVariableOp_3ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_3Ј
1GRU_classifier/gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   23
1GRU_classifier/gru/gru_cell/strided_slice_2/stackї
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell/strided_slice_2/stack_1ї
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
:	ђ *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_2е
$GRU_classifier/gru/gru_cell/MatMul_2MatMul%GRU_classifier/gru/gru_cell/mul_2:z:04GRU_classifier/gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$GRU_classifier/gru/gru_cell/MatMul_2∞
1GRU_classifier/gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_3/stackі
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_1і
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_3/stack_2Ф
+GRU_classifier/gru/gru_cell/strided_slice_3StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:0:GRU_classifier/gru/gru_cell/strided_slice_3/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_3/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_3л
#GRU_classifier/gru/gru_cell/BiasAddBiasAdd,GRU_classifier/gru/gru_cell/MatMul:product:04GRU_classifier/gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#GRU_classifier/gru/gru_cell/BiasAdd∞
1GRU_classifier/gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_4/stackі
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_1і
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_4/stack_2В
+GRU_classifier/gru/gru_cell/strided_slice_4StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:0:GRU_classifier/gru/gru_cell/strided_slice_4/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_4/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2-
+GRU_classifier/gru/gru_cell/strided_slice_4с
%GRU_classifier/gru/gru_cell/BiasAdd_1BiasAdd.GRU_classifier/gru/gru_cell/MatMul_1:product:04GRU_classifier/gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_1∞
1GRU_classifier/gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@23
1GRU_classifier/gru/gru_cell/strided_slice_5/stackі
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_1і
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_5/stack_2Т
+GRU_classifier/gru/gru_cell/strided_slice_5StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:0:GRU_classifier/gru/gru_cell/strided_slice_5/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_5/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_5с
%GRU_classifier/gru/gru_cell/BiasAdd_2BiasAdd.GRU_classifier/gru/gru_cell/MatMul_2:product:04GRU_classifier/gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_2‘
!GRU_classifier/gru/gru_cell/mul_3Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/mul_3‘
!GRU_classifier/gru/gru_cell/mul_4Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/mul_4‘
!GRU_classifier/gru/gru_cell/mul_5Mul!GRU_classifier/gru/zeros:output:00GRU_classifier/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/mul_5“
,GRU_classifier/gru/gru_cell/ReadVariableOp_4ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_4Ј
1GRU_classifier/gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_6/stackї
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_1ї
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_6/stack_2∞
+GRU_classifier/gru/gru_cell/strided_slice_6StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_4:value:0:GRU_classifier/gru/gru_cell/strided_slice_6/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_6/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_6е
$GRU_classifier/gru/gru_cell/MatMul_3MatMul%GRU_classifier/gru/gru_cell/mul_3:z:04GRU_classifier/gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$GRU_classifier/gru/gru_cell/MatMul_3“
,GRU_classifier/gru/gru_cell/ReadVariableOp_5ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_5Ј
1GRU_classifier/gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1GRU_classifier/gru/gru_cell/strided_slice_7/stackї
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   25
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_1ї
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3GRU_classifier/gru/gru_cell/strided_slice_7/stack_2∞
+GRU_classifier/gru/gru_cell/strided_slice_7StridedSlice4GRU_classifier/gru/gru_cell/ReadVariableOp_5:value:0:GRU_classifier/gru/gru_cell/strided_slice_7/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_7/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_7е
$GRU_classifier/gru/gru_cell/MatMul_4MatMul%GRU_classifier/gru/gru_cell/mul_4:z:04GRU_classifier/gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$GRU_classifier/gru/gru_cell/MatMul_4∞
1GRU_classifier/gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_8/stackі
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_1і
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_8/stack_2Ф
+GRU_classifier/gru/gru_cell/strided_slice_8StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:1:GRU_classifier/gru/gru_cell/strided_slice_8/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_8/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2-
+GRU_classifier/gru/gru_cell/strided_slice_8с
%GRU_classifier/gru/gru_cell/BiasAdd_3BiasAdd.GRU_classifier/gru/gru_cell/MatMul_3:product:04GRU_classifier/gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_3∞
1GRU_classifier/gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1GRU_classifier/gru/gru_cell/strided_slice_9/stackі
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@25
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_1і
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3GRU_classifier/gru/gru_cell/strided_slice_9/stack_2В
+GRU_classifier/gru/gru_cell/strided_slice_9StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:1:GRU_classifier/gru/gru_cell/strided_slice_9/stack:output:0<GRU_classifier/gru/gru_cell/strided_slice_9/stack_1:output:0<GRU_classifier/gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2-
+GRU_classifier/gru/gru_cell/strided_slice_9с
%GRU_classifier/gru/gru_cell/BiasAdd_4BiasAdd.GRU_classifier/gru/gru_cell/MatMul_4:product:04GRU_classifier/gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_4џ
GRU_classifier/gru/gru_cell/addAddV2,GRU_classifier/gru/gru_cell/BiasAdd:output:0.GRU_classifier/gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
GRU_classifier/gru/gru_cell/addђ
#GRU_classifier/gru/gru_cell/SigmoidSigmoid#GRU_classifier/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#GRU_classifier/gru/gru_cell/Sigmoidб
!GRU_classifier/gru/gru_cell/add_1AddV2.GRU_classifier/gru/gru_cell/BiasAdd_1:output:0.GRU_classifier/gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/add_1≤
%GRU_classifier/gru/gru_cell/Sigmoid_1Sigmoid%GRU_classifier/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/gru_cell/Sigmoid_1“
,GRU_classifier/gru/gru_cell/ReadVariableOp_6ReadVariableOp5gru_classifier_gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02.
,GRU_classifier/gru/gru_cell/ReadVariableOp_6є
2GRU_classifier/gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   24
2GRU_classifier/gru/gru_cell/strided_slice_10/stackљ
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4GRU_classifier/gru/gru_cell/strided_slice_10/stack_1љ
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
,GRU_classifier/gru/gru_cell/strided_slice_10ж
$GRU_classifier/gru/gru_cell/MatMul_5MatMul%GRU_classifier/gru/gru_cell/mul_5:z:05GRU_classifier/gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$GRU_classifier/gru/gru_cell/MatMul_5≤
2GRU_classifier/gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@24
2GRU_classifier/gru/gru_cell/strided_slice_11/stackґ
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_1ґ
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4GRU_classifier/gru/gru_cell/strided_slice_11/stack_2Ч
,GRU_classifier/gru/gru_cell/strided_slice_11StridedSlice,GRU_classifier/gru/gru_cell/unstack:output:1;GRU_classifier/gru/gru_cell/strided_slice_11/stack:output:0=GRU_classifier/gru/gru_cell/strided_slice_11/stack_1:output:0=GRU_classifier/gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2.
,GRU_classifier/gru/gru_cell/strided_slice_11т
%GRU_classifier/gru/gru_cell/BiasAdd_5BiasAdd.GRU_classifier/gru/gru_cell/MatMul_5:product:05GRU_classifier/gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/gru_cell/BiasAdd_5Џ
!GRU_classifier/gru/gru_cell/mul_6Mul)GRU_classifier/gru/gru_cell/Sigmoid_1:y:0.GRU_classifier/gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/mul_6Ў
!GRU_classifier/gru/gru_cell/add_2AddV2.GRU_classifier/gru/gru_cell/BiasAdd_2:output:0%GRU_classifier/gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/add_2•
 GRU_classifier/gru/gru_cell/TanhTanh%GRU_classifier/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 GRU_classifier/gru/gru_cell/TanhЋ
!GRU_classifier/gru/gru_cell/mul_7Mul'GRU_classifier/gru/gru_cell/Sigmoid:y:0!GRU_classifier/gru/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/mul_7Л
!GRU_classifier/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!GRU_classifier/gru/gru_cell/sub/x–
GRU_classifier/gru/gru_cell/subSub*GRU_classifier/gru/gru_cell/sub/x:output:0'GRU_classifier/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
GRU_classifier/gru/gru_cell/sub 
!GRU_classifier/gru/gru_cell/mul_8Mul#GRU_classifier/gru/gru_cell/sub:z:0$GRU_classifier/gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/mul_8ѕ
!GRU_classifier/gru/gru_cell/add_3AddV2%GRU_classifier/gru/gru_cell/mul_7:z:0%GRU_classifier/gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/gru_cell/add_3µ
0GRU_classifier/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0GRU_classifier/gru/TensorArrayV2_1/element_shapeД
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
GRU_classifier/gru/timeѓ
0GRU_classifier/gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€22
0GRU_classifier/gru/TensorArrayV2_2/element_shapeД
"GRU_classifier/gru/TensorArrayV2_2TensorListReserve9GRU_classifier/gru/TensorArrayV2_2/element_shape:output:0+GRU_classifier/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02$
"GRU_classifier/gru/TensorArrayV2_2й
JGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2L
JGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeћ
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor"GRU_classifier/gru/transpose_1:y:0SGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02>
<GRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor§
GRU_classifier/gru/zeros_like	ZerosLike%GRU_classifier/gru/gru_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
GRU_classifier/gru/zeros_like•
+GRU_classifier/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2-
+GRU_classifier/gru/while/maximum_iterationsР
%GRU_classifier/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%GRU_classifier/gru/while/loop_counterЄ
GRU_classifier/gru/whileWhile.GRU_classifier/gru/while/loop_counter:output:04GRU_classifier/gru/while/maximum_iterations:output:0 GRU_classifier/gru/time:output:0+GRU_classifier/gru/TensorArrayV2_1:handle:0!GRU_classifier/gru/zeros_like:y:0!GRU_classifier/gru/zeros:output:0+GRU_classifier/gru/strided_slice_1:output:0JGRU_classifier/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0LGRU_classifier/gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:03gru_classifier_gru_gru_cell_readvariableop_resource5gru_classifier_gru_gru_cell_readvariableop_1_resource5gru_classifier_gru_gru_cell_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"GRU_classifier_gru_while_body_8756*.
cond&R$
"GRU_classifier_gru_while_cond_8755*M
output_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : *
parallel_iterations 2
GRU_classifier/gru/whileџ
CGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2E
CGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shapeљ
5GRU_classifier/gru/TensorArrayV2Stack/TensorListStackTensorListStack!GRU_classifier/gru/while:output:3LGRU_classifier/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype027
5GRU_classifier/gru/TensorArrayV2Stack/TensorListStackІ
(GRU_classifier/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2*
(GRU_classifier/gru/strided_slice_3/stackҐ
*GRU_classifier/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*GRU_classifier/gru/strided_slice_3/stack_1Ґ
*GRU_classifier/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*GRU_classifier/gru/strided_slice_3/stack_2М
"GRU_classifier/gru/strided_slice_3StridedSlice>GRU_classifier/gru/TensorArrayV2Stack/TensorListStack:tensor:01GRU_classifier/gru/strided_slice_3/stack:output:03GRU_classifier/gru/strided_slice_3/stack_1:output:03GRU_classifier/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2$
"GRU_classifier/gru/strided_slice_3Я
#GRU_classifier/gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#GRU_classifier/gru/transpose_2/permъ
GRU_classifier/gru/transpose_2	Transpose>GRU_classifier/gru/TensorArrayV2Stack/TensorListStack:tensor:0,GRU_classifier/gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2 
GRU_classifier/gru/transpose_2М
GRU_classifier/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_classifier/gru/runtimeЎ
.GRU_classifier/output/Tensordot/ReadVariableOpReadVariableOp7gru_classifier_output_tensordot_readvariableop_resource*
_output_shapes

: *
dtype020
.GRU_classifier/output/Tensordot/ReadVariableOpЦ
$GRU_classifier/output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$GRU_classifier/output/Tensordot/axesЭ
$GRU_classifier/output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$GRU_classifier/output/Tensordot/free†
%GRU_classifier/output/Tensordot/ShapeShape"GRU_classifier/gru/transpose_2:y:0*
T0*
_output_shapes
:2'
%GRU_classifier/output/Tensordot/Shape†
-GRU_classifier/output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-GRU_classifier/output/Tensordot/GatherV2/axisњ
(GRU_classifier/output/Tensordot/GatherV2GatherV2.GRU_classifier/output/Tensordot/Shape:output:0-GRU_classifier/output/Tensordot/free:output:06GRU_classifier/output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(GRU_classifier/output/Tensordot/GatherV2§
/GRU_classifier/output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/GRU_classifier/output/Tensordot/GatherV2_1/axis≈
*GRU_classifier/output/Tensordot/GatherV2_1GatherV2.GRU_classifier/output/Tensordot/Shape:output:0-GRU_classifier/output/Tensordot/axes:output:08GRU_classifier/output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*GRU_classifier/output/Tensordot/GatherV2_1Ш
%GRU_classifier/output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%GRU_classifier/output/Tensordot/ConstЎ
$GRU_classifier/output/Tensordot/ProdProd1GRU_classifier/output/Tensordot/GatherV2:output:0.GRU_classifier/output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$GRU_classifier/output/Tensordot/ProdЬ
'GRU_classifier/output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'GRU_classifier/output/Tensordot/Const_1а
&GRU_classifier/output/Tensordot/Prod_1Prod3GRU_classifier/output/Tensordot/GatherV2_1:output:00GRU_classifier/output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&GRU_classifier/output/Tensordot/Prod_1Ь
+GRU_classifier/output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+GRU_classifier/output/Tensordot/concat/axisЮ
&GRU_classifier/output/Tensordot/concatConcatV2-GRU_classifier/output/Tensordot/free:output:0-GRU_classifier/output/Tensordot/axes:output:04GRU_classifier/output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&GRU_classifier/output/Tensordot/concatд
%GRU_classifier/output/Tensordot/stackPack-GRU_classifier/output/Tensordot/Prod:output:0/GRU_classifier/output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%GRU_classifier/output/Tensordot/stackч
)GRU_classifier/output/Tensordot/transpose	Transpose"GRU_classifier/gru/transpose_2:y:0/GRU_classifier/output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2+
)GRU_classifier/output/Tensordot/transposeч
'GRU_classifier/output/Tensordot/ReshapeReshape-GRU_classifier/output/Tensordot/transpose:y:0.GRU_classifier/output/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'GRU_classifier/output/Tensordot/Reshapeц
&GRU_classifier/output/Tensordot/MatMulMatMul0GRU_classifier/output/Tensordot/Reshape:output:06GRU_classifier/output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2(
&GRU_classifier/output/Tensordot/MatMulЬ
'GRU_classifier/output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'GRU_classifier/output/Tensordot/Const_2†
-GRU_classifier/output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-GRU_classifier/output/Tensordot/concat_1/axisЂ
(GRU_classifier/output/Tensordot/concat_1ConcatV21GRU_classifier/output/Tensordot/GatherV2:output:00GRU_classifier/output/Tensordot/Const_2:output:06GRU_classifier/output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(GRU_classifier/output/Tensordot/concat_1с
GRU_classifier/output/TensordotReshape0GRU_classifier/output/Tensordot/MatMul:product:01GRU_classifier/output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2!
GRU_classifier/output/Tensordotќ
,GRU_classifier/output/BiasAdd/ReadVariableOpReadVariableOp5gru_classifier_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,GRU_classifier/output/BiasAdd/ReadVariableOpи
GRU_classifier/output/BiasAddBiasAdd(GRU_classifier/output/Tensordot:output:04GRU_classifier/output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
GRU_classifier/output/BiasAdd…
IdentityIdentity&GRU_classifier/output/BiasAdd:output:0+^GRU_classifier/gru/gru_cell/ReadVariableOp-^GRU_classifier/gru/gru_cell/ReadVariableOp_1-^GRU_classifier/gru/gru_cell/ReadVariableOp_2-^GRU_classifier/gru/gru_cell/ReadVariableOp_3-^GRU_classifier/gru/gru_cell/ReadVariableOp_4-^GRU_classifier/gru/gru_cell/ReadVariableOp_5-^GRU_classifier/gru/gru_cell/ReadVariableOp_6^GRU_classifier/gru/while-^GRU_classifier/output/BiasAdd/ReadVariableOp/^GRU_classifier/output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2X
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
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_nameinput
≤
ћ
__inference_loss_fn_1_13408Z
Hgru_gru_cell_recurrent_kernel_regularizer_square_readvariableop_resource: `
identityИҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpЛ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpHgru_gru_cell_recurrent_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulґ
IdentityIdentity1gru/gru_cell/recurrent_kernel/Regularizer/mul:z:0@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
¬ 
ш
A__inference_output_layer_call_and_return_conditional_losses_10129

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЩ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Щ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpР
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2	
BiasAdd•
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
§ї
А
B__inference_gru_cell_layer_call_and_return_conditional_losses_9376

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	ђ`+
readvariableop_4_resource: `
identity

identity_1ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3ҐReadVariableOp_4ҐReadVariableOp_5ҐReadVariableOp_6Ґ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
 *  А?2
ones_like/ConstЕ
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout/ConstА
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape—
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ЯТЊ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_1/ConstЖ
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape„
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2§Ы–2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_1/GreaterEqual/y«
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/GreaterEqualЖ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/CastГ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_2/ConstЖ
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape„
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2їБ„2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_2/GreaterEqual/y«
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_2/GreaterEqualЖ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
dropout_2/CastГ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
ones_like_1/ConstМ
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ones_like_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_3/ConstЗ
dropout_3/MulMulones_like_1:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/Shape÷
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ёХ≤2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_3/GreaterEqual/y∆
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/Mul_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_4/ConstЗ
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape÷
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ЅЪз2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_4/GreaterEqual/y∆
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/GreaterEqualЕ
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/CastВ
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_5/ConstЗ
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape÷
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2б—¶2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_5/GreaterEqual/y∆
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/GreaterEqualЕ
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/CastВ
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ђ2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
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
strided_slice/stack_2€
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Й
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2Й
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
strided_slice_3/stack_2м
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
:€€€€€€€€€ 2	
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
strided_slice_4/stack_2Џ
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4Б
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_5/stack_2к
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5Б
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_8/stack_2м
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8Б
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_9/stack_2Џ
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9Б
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6Б
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2Н
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
:€€€€€€€€€ 2

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
strided_slice_11/stack_2п
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11В
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_3…
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul№
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЏ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
А
і
#__inference_gru_layer_call_fn_13009
inputs_0
unknown:`
	unknown_0:	ђ`
	unknown_1: `
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_95192
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
Ъќ
¬
>__inference_gru_layer_call_and_return_conditional_losses_12596

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	ђ`4
"gru_cell_readvariableop_4_resource: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐgru_cell/ReadVariableOpҐgru_cell/ReadVariableOp_1Ґgru_cell/ReadVariableOp_2Ґgru_cell/ReadVariableOp_3Ґgru_cell/ReadVariableOp_4Ґgru_cell/ReadVariableOp_5Ґgru_cell/ReadVariableOp_6ҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
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
 *  А?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
gru_cell/ones_like_1/Const∞
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/ones_like_1У
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOpЕ
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstackН
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mulС
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_1С
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_2Ъ
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_1Н
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stackС
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1С
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
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_sliceС
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMulЪ
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_2С
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stackХ
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1Х
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2њ
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_1Щ
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_1Ъ
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_3С
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stackХ
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1Х
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2њ
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_2Щ
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_2К
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stackО
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1О
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2Ґ
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3Я
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAddК
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stackО
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1О
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2Р
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4•
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_1К
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stackО
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1О
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2†
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5•
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_2И
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_3И
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_4И
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_5Щ
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4С
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stackХ
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1Х
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2Њ
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6Щ
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_3Щ
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5С
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stackХ
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1Х
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2Њ
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7Щ
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_4К
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stackО
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1О
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2Ґ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8•
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_3К
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stackО
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1О
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2Р
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9•
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_4П
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/SigmoidХ
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Sigmoid_1Щ
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6У
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stackЧ
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1Ч
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2√
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10Ъ
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_5М
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stackР
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1Р
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2•
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¶
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_5О
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_6М
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru_cell/sub/xД
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_8Г
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_12432*
condR
while_cond_12431*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime“
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulе
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulі
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
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
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
Ђ

’
(__inference_gru_cell_layer_call_fn_13386

inputs
states_0
unknown:`
	unknown_0:	ђ`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_93762
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
¬ 
ш
A__inference_output_layer_call_and_return_conditional_losses_13061

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЩ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Щ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpР
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2	
BiasAdd•
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
к	
]
A__inference_masking_layer_call_and_return_conditional_losses_9795

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
!:€€€€€€€€€€€€€€€€€€ђ2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Any/reduction_indicesЖ
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
mulЖ
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
squeeze_dims

€€€€€€€€€2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€€€€€€€€€€ђ:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
х
•
while_cond_12088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_12088___redundant_placeholder03
/while_while_cond_12088___redundant_placeholder13
/while_while_cond_12088___redundant_placeholder23
/while_while_cond_12088___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ќ
У
&__inference_output_layer_call_fn_13070

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_101292
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ђ

’
(__inference_gru_cell_layer_call_fn_13372

inputs
states_0
unknown:`
	unknown_0:	ђ`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_90982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
х
•
while_cond_12774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_12774___redundant_placeholder03
/while_while_cond_12774___redundant_placeholder13
/while_while_cond_12774___redundant_placeholder23
/while_while_cond_12774___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
©≤
Ћ
while_body_9927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	ђ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	ђ`:
(while_gru_cell_readvariableop_4_resource: `ИҐwhile/gru_cell/ReadVariableOpҐwhile/gru_cell/ReadVariableOp_1Ґwhile/gru_cell/ReadVariableOp_2Ґwhile/gru_cell/ReadVariableOp_3Ґwhile/gru_cell/ReadVariableOp_4Ґwhile/gru_cell/ReadVariableOp_5Ґwhile/gru_cell/ReadVariableOp_6√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem†
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/ShapeЕ
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
while/gru_cell/ones_like/ConstЅ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/ones_likeЗ
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/ShapeЙ
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 while/gru_cell/ones_like_1/Const»
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/ones_like_1І
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOpЧ
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstackЈ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mulї
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_1ї
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_2Ѓ
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_1Щ
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stackЭ
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1Э
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2ў
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMulЃ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_2Э
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack°
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1°
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2г
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_1Ѓ
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_3Э
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack°
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1°
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2г
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_2Ц
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stackЪ
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1Ъ
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2∆
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3Ј
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAddЦ
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stackЪ
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1Ъ
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2і
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4љ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_1Ц
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stackЪ
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1Ъ
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2ƒ
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5љ
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_2Я
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_3Я
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_4Я
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_5≠
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4Э
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack°
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1°
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_3≠
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5Э
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack°
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1°
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_4Ц
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stackЪ
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1Ъ
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2∆
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8љ
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_3Ц
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stackЪ
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1Ъ
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2і
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9љ
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_4І
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/addЕ
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid≠
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_1Л
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid_1≠
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6Я
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
'while/gru_cell/strided_slice_10/stack_2з
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10≤
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_5Ш
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stackЬ
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1Ь
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2…
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11Њ
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_5¶
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_6§
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/TanhЦ
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/gru_cell/sub/xЬ
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/subЦ
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_8Ы
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_3№
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
while/add_1 
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityЁ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1ћ
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3и
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2>
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Л'
і
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10693	
input
	gru_10668:`
	gru_10670:	ђ`
	gru_10672: `
output_10675: 
output_10677:
identityИҐgru/StatefulPartitionedCallҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐoutput/StatefulPartitionedCallб
masking/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_masking_layer_call_and_return_conditional_losses_97952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_10668	gru_10670	gru_10672*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_100912
gru/StatefulPartitionedCallЈ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_10675output_10677*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_101292 
output/StatefulPartitionedCallє
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10670*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulћ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10672*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_nameinput
е
у
.__inference_GRU_classifier_layer_call_fn_11572

inputs
unknown:`
	unknown_0:	ђ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8В *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_101482
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
чщ
ћ
while_body_12089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	ђ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	ђ`:
(while_gru_cell_readvariableop_4_resource: `ИҐwhile/gru_cell/ReadVariableOpҐwhile/gru_cell/ReadVariableOp_1Ґwhile/gru_cell/ReadVariableOp_2Ґwhile/gru_cell/ReadVariableOp_3Ґwhile/gru_cell/ReadVariableOp_4Ґwhile/gru_cell/ReadVariableOp_5Ґwhile/gru_cell/ReadVariableOp_6√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem†
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/ShapeЕ
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
while/gru_cell/ones_like/ConstЅ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/ones_likeБ
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
while/gru_cell/dropout/ConstЉ
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/MulН
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeю
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2„ѕ№25
3while/gru_cell/dropout/random_uniform/RandomUniformУ
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%while/gru_cell/dropout/GreaterEqual/yы
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2%
#while/gru_cell/dropout/GreaterEqual≠
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/CastЈ
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/Mul_1Е
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_1/Const¬
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_1/MulС
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/ShapeД
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2≥ќК27
5while/gru_cell/dropout_1/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_1/GreaterEqual/yГ
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%while/gru_cell/dropout_1/GreaterEqual≥
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_1/Castњ
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
while/gru_cell/dropout_1/Mul_1Е
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_2/Const¬
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_2/MulС
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/ShapeД
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ўР‘27
5while/gru_cell/dropout_2/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_2/GreaterEqual/yГ
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%while/gru_cell/dropout_2/GreaterEqual≥
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_2/Castњ
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
while/gru_cell/dropout_2/Mul_1З
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/ShapeЙ
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 while/gru_cell/ones_like_1/Const»
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/ones_like_1Е
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_3/Const√
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_3/MulУ
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/ShapeГ
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ќю„27
5while/gru_cell/dropout_3/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_3/GreaterEqual/yВ
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_3/GreaterEqual≤
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_3/CastЊ
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_3/Mul_1Е
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_4/Const√
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_4/MulУ
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/ShapeГ
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2°љ†27
5while/gru_cell/dropout_4/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_4/GreaterEqual/yВ
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_4/GreaterEqual≤
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_4/CastЊ
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_4/Mul_1Е
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_5/Const√
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_5/MulУ
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/ShapeГ
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ж≤Ђ27
5while/gru_cell/dropout_5/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_5/GreaterEqual/yВ
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_5/GreaterEqual≤
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_5/CastЊ
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_5/Mul_1І
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOpЧ
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstackґ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mulЉ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_1Љ
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_2Ѓ
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_1Щ
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stackЭ
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1Э
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2ў
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMulЃ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_2Э
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack°
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1°
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2г
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_1Ѓ
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_3Э
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack°
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1°
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2г
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_2Ц
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stackЪ
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1Ъ
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2∆
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3Ј
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAddЦ
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stackЪ
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1Ъ
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2і
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4љ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_1Ц
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stackЪ
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1Ъ
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2ƒ
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5љ
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_2Ю
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_3Ю
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_4Ю
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_5≠
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4Э
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack°
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1°
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_3≠
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5Э
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack°
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1°
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_4Ц
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stackЪ
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1Ъ
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2∆
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8љ
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_3Ц
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stackЪ
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1Ъ
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2і
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9љ
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_4І
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/addЕ
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid≠
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_1Л
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid_1≠
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6Я
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
'while/gru_cell/strided_slice_10/stack_2з
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10≤
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_5Ш
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stackЬ
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1Ь
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2…
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11Њ
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_5¶
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_6§
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/TanhЦ
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/gru_cell/sub/xЬ
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/subЦ
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_8Ы
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_3№
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
while/add_1 
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityЁ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1ћ
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3и
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2>
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
цщ
ћ
while_body_10363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	ђ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	ђ`:
(while_gru_cell_readvariableop_4_resource: `ИҐwhile/gru_cell/ReadVariableOpҐwhile/gru_cell/ReadVariableOp_1Ґwhile/gru_cell/ReadVariableOp_2Ґwhile/gru_cell/ReadVariableOp_3Ґwhile/gru_cell/ReadVariableOp_4Ґwhile/gru_cell/ReadVariableOp_5Ґwhile/gru_cell/ReadVariableOp_6√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem†
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/ShapeЕ
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
while/gru_cell/ones_like/ConstЅ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/ones_likeБ
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
while/gru_cell/dropout/ConstЉ
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/MulН
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeю
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ҐэС25
3while/gru_cell/dropout/random_uniform/RandomUniformУ
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%while/gru_cell/dropout/GreaterEqual/yы
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2%
#while/gru_cell/dropout/GreaterEqual≠
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/CastЈ
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/Mul_1Е
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_1/Const¬
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_1/MulС
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/ShapeД
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2Щ÷ї27
5while/gru_cell/dropout_1/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_1/GreaterEqual/yГ
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%while/gru_cell/dropout_1/GreaterEqual≥
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_1/Castњ
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
while/gru_cell/dropout_1/Mul_1Е
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_2/Const¬
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_2/MulС
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/ShapeД
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2≥ня27
5while/gru_cell/dropout_2/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_2/GreaterEqual/yГ
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%while/gru_cell/dropout_2/GreaterEqual≥
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_2/Castњ
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
while/gru_cell/dropout_2/Mul_1З
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/ShapeЙ
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 while/gru_cell/ones_like_1/Const»
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/ones_like_1Е
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_3/Const√
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_3/MulУ
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/ShapeГ
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ајв27
5while/gru_cell/dropout_3/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_3/GreaterEqual/yВ
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_3/GreaterEqual≤
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_3/CastЊ
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_3/Mul_1Е
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_4/Const√
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_4/MulУ
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/ShapeГ
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2йуУ27
5while/gru_cell/dropout_4/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_4/GreaterEqual/yВ
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_4/GreaterEqual≤
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_4/CastЊ
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_4/Mul_1Е
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_5/Const√
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_5/MulУ
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/ShapeВ
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ЭєI27
5while/gru_cell/dropout_5/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_5/GreaterEqual/yВ
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_5/GreaterEqual≤
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_5/CastЊ
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_5/Mul_1І
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOpЧ
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstackґ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mulЉ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_1Љ
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_2Ѓ
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_1Щ
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stackЭ
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1Э
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2ў
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMulЃ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_2Э
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack°
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1°
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2г
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_1Ѓ
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_3Э
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack°
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1°
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2г
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_2Ц
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stackЪ
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1Ъ
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2∆
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3Ј
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAddЦ
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stackЪ
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1Ъ
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2і
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4љ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_1Ц
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stackЪ
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1Ъ
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2ƒ
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5љ
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_2Ю
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_3Ю
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_4Ю
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_5≠
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4Э
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack°
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1°
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_3≠
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5Э
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack°
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1°
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_4Ц
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stackЪ
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1Ъ
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2∆
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8љ
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_3Ц
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stackЪ
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1Ъ
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2і
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9љ
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_4І
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/addЕ
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid≠
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_1Л
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid_1≠
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6Я
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
'while/gru_cell/strided_slice_10/stack_2з
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10≤
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_5Ш
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stackЬ
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1Ь
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2…
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11Њ
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_5¶
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_6§
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/TanhЦ
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/gru_cell/sub/xЬ
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/subЦ
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_8Ы
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_3№
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
while/add_1 
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityЁ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1ћ
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3и
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2>
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
™≤
ћ
while_body_12432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	ђ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	ђ`:
(while_gru_cell_readvariableop_4_resource: `ИҐwhile/gru_cell/ReadVariableOpҐwhile/gru_cell/ReadVariableOp_1Ґwhile/gru_cell/ReadVariableOp_2Ґwhile/gru_cell/ReadVariableOp_3Ґwhile/gru_cell/ReadVariableOp_4Ґwhile/gru_cell/ReadVariableOp_5Ґwhile/gru_cell/ReadVariableOp_6√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem†
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/ShapeЕ
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
while/gru_cell/ones_like/ConstЅ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/ones_likeЗ
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/ShapeЙ
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 while/gru_cell/ones_like_1/Const»
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/ones_like_1І
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOpЧ
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstackЈ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mulї
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_1ї
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_2Ѓ
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_1Щ
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stackЭ
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1Э
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2ў
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMulЃ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_2Э
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack°
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1°
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2г
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_1Ѓ
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_3Э
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack°
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1°
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2г
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_2Ц
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stackЪ
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1Ъ
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2∆
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3Ј
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAddЦ
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stackЪ
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1Ъ
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2і
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4љ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_1Ц
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stackЪ
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1Ъ
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2ƒ
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5љ
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_2Я
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_3Я
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_4Я
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_5≠
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4Э
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack°
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1°
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_3≠
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5Э
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack°
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1°
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_4Ц
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stackЪ
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1Ъ
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2∆
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8љ
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_3Ц
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stackЪ
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1Ъ
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2і
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9љ
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_4І
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/addЕ
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid≠
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_1Л
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid_1≠
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6Я
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
'while/gru_cell/strided_slice_10/stack_2з
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10≤
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_5Ш
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stackЬ
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1Ь
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2…
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11Њ
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_5¶
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_6§
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/TanhЦ
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/gru_cell/sub/xЬ
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/subЦ
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_8Ы
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_3№
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
while/add_1 
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityЁ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1ћ
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3и
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2>
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
цщ
ћ
while_body_12775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	ђ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	ђ`:
(while_gru_cell_readvariableop_4_resource: `ИҐwhile/gru_cell/ReadVariableOpҐwhile/gru_cell/ReadVariableOp_1Ґwhile/gru_cell/ReadVariableOp_2Ґwhile/gru_cell/ReadVariableOp_3Ґwhile/gru_cell/ReadVariableOp_4Ґwhile/gru_cell/ReadVariableOp_5Ґwhile/gru_cell/ReadVariableOp_6√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem†
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/ShapeЕ
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
while/gru_cell/ones_like/ConstЅ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/ones_likeБ
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
while/gru_cell/dropout/ConstЉ
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/MulН
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shapeю
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ЌИ‘25
3while/gru_cell/dropout/random_uniform/RandomUniformУ
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%while/gru_cell/dropout/GreaterEqual/yы
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2%
#while/gru_cell/dropout/GreaterEqual≠
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/CastЈ
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout/Mul_1Е
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_1/Const¬
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_1/MulС
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/ShapeД
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2†єД27
5while/gru_cell/dropout_1/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_1/GreaterEqual/yГ
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%while/gru_cell/dropout_1/GreaterEqual≥
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_1/Castњ
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
while/gru_cell/dropout_1/Mul_1Е
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_2/Const¬
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_2/MulС
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/ShapeД
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2я№љ27
5while/gru_cell/dropout_2/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_2/GreaterEqual/yГ
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%while/gru_cell/dropout_2/GreaterEqual≥
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/dropout_2/Castњ
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
while/gru_cell/dropout_2/Mul_1З
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/ShapeЙ
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 while/gru_cell/ones_like_1/Const»
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/ones_like_1Е
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_3/Const√
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_3/MulУ
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/ShapeГ
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2Ѕџј27
5while/gru_cell/dropout_3/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_3/GreaterEqual/yВ
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_3/GreaterEqual≤
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_3/CastЊ
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_3/Mul_1Е
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_4/Const√
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_4/MulУ
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/ShapeГ
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2„Не27
5while/gru_cell/dropout_4/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_4/GreaterEqual/yВ
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_4/GreaterEqual≤
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_4/CastЊ
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_4/Mul_1Е
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2 
while/gru_cell/dropout_5/Const√
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_5/MulУ
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/ShapeВ
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2£ИS27
5while/gru_cell/dropout_5/random_uniform/RandomUniformЧ
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2)
'while/gru_cell/dropout_5/GreaterEqual/yВ
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%while/gru_cell/dropout_5/GreaterEqual≤
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/dropout_5/CastЊ
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
while/gru_cell/dropout_5/Mul_1І
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOpЧ
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstackґ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mulЉ
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_1Љ
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_2Ѓ
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_1Щ
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stackЭ
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1Э
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2ў
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMulЃ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_2Э
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack°
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1°
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2г
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_1Ѓ
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_3Э
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack°
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1°
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2г
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_2Ц
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stackЪ
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1Ъ
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2∆
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3Ј
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAddЦ
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stackЪ
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1Ъ
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2і
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4љ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_1Ц
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stackЪ
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1Ъ
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2ƒ
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5љ
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_2Ю
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_3Ю
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_4Ю
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_5≠
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4Э
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack°
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1°
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_3≠
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5Э
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack°
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1°
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_4Ц
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stackЪ
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1Ъ
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2∆
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8љ
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_3Ц
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stackЪ
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1Ъ
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2і
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9љ
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_4І
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/addЕ
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid≠
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_1Л
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid_1≠
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6Я
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
'while/gru_cell/strided_slice_10/stack_2з
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10≤
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_5Ш
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stackЬ
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1Ь
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2…
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11Њ
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_5¶
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_6§
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/TanhЦ
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/gru_cell/sub/xЬ
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/subЦ
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_8Ы
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_3№
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
while/add_1 
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityЁ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1ћ
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3и
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2>
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ы
≤
#__inference_gru_layer_call_fn_13031

inputs
unknown:`
	unknown_0:	ђ`
	unknown_1: `
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_105752
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
р
†
while_cond_9110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9110___redundant_placeholder02
.while_while_cond_9110___redundant_placeholder12
.while_while_cond_9110___redundant_placeholder22
.while_while_cond_9110___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
¶Ф
≠
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_11109

inputs6
$gru_gru_cell_readvariableop_resource:`9
&gru_gru_cell_readvariableop_1_resource:	ђ`8
&gru_gru_cell_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identityИҐgru/gru_cell/ReadVariableOpҐgru/gru_cell/ReadVariableOp_1Ґgru/gru_cell/ReadVariableOp_2Ґgru/gru_cell/ReadVariableOp_3Ґgru/gru_cell/ReadVariableOp_4Ґgru/gru_cell/ReadVariableOp_5Ґgru/gru_cell/ReadVariableOp_6Ґ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ	gru/whileҐoutput/BiasAdd/ReadVariableOpҐoutput/Tensordot/ReadVariableOpm
masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking/NotEqual/yХ
masking/NotEqualNotEqualinputsmasking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
masking/NotEqualЙ
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
masking/Any/reduction_indices¶
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
	keep_dims(2
masking/AnyИ
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
masking/mulЮ
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
squeeze_dims

€€€€€€€€€2
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
gru/strided_slice/stackА
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1А
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ъ
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
B :и2
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
gru/zeros/packed/1У
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
gru/zeros/ConstЕ
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/permЩ
gru/transpose	Transposemasking/mul:z:0gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1А
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stackД
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1Д
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2Ж
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
€€€€€€€€€2
gru/ExpandDims/dim§
gru/ExpandDims
ExpandDimsmasking/Squeeze:output:0gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
gru/ExpandDimsБ
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm¶
gru/transpose_1	Transposegru/ExpandDims:output:0gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
gru/transpose_1Н
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
gru/TensorArrayV2/element_shape¬
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2«
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensorА
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stackД
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1Д
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2Х
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_mask2
gru/strided_slice_2И
gru/gru_cell/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/ShapeБ
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru/gru_cell/ones_like/Constє
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/ones_likeВ
gru/gru_cell/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2 
gru/gru_cell/ones_like_1/ShapeЕ
gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
gru/gru_cell/ones_like_1/Constј
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/ones_like_1Я
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell/ReadVariableOpС
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell/unstackЭ
gru/gru_cell/mulMulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/mul°
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/mul_1°
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/mul_2¶
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru/gru_cell/ReadVariableOp_1Х
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stackЩ
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice/stack_1Щ
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2Ќ
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice°
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul¶
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru/gru_cell/ReadVariableOp_2Щ
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_1/stackЭ
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_1/stack_1Э
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2„
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1©
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul_1¶
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru/gru_cell/ReadVariableOp_3Щ
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru/gru_cell/strided_slice_2/stackЭ
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1Э
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2„
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2©
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul_2Т
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stackЦ
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_3/stack_1Ц
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2Ї
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_3ѓ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAddТ
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_4/stackЦ
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_4/stack_1Ц
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2®
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_4µ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_1Т
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2$
"gru/gru_cell/strided_slice_5/stackЦ
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1Ц
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2Є
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
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_2Ш
gru/gru_cell/mul_3Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_3Ш
gru/gru_cell/mul_4Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_4Ш
gru/gru_cell/mul_5Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_5•
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_4Щ
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stackЭ
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_6/stack_1Э
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2÷
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
:€€€€€€€€€ 2
gru/gru_cell/MatMul_3•
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_5Щ
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_7/stackЭ
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_7/stack_1Э
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2÷
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
:€€€€€€€€€ 2
gru/gru_cell/MatMul_4Т
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stackЦ
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_8/stack_1Ц
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2Ї
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
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_3Т
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_9/stackЦ
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_9/stack_1Ц
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2®
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_9µ
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_4Я
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/Sigmoid•
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add_1Е
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/Sigmoid_1•
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_6Ы
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#gru/gru_cell/strided_slice_10/stackЯ
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1Я
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2џ
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10™
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul_5Ф
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2%
#gru/gru_cell/strided_slice_11/stackШ
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1Ш
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2љ
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_11ґ
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_5Ю
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_6Ь
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/TanhП
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_7m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru/gru_cell/sub/xФ
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/subО
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_8У
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add_3Ч
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2#
!gru/TensorArrayV2_1/element_shape»
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

gru/timeС
!gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2#
!gru/TensorArrayV2_2/element_shape»
gru/TensorArrayV2_2TensorListReserve*gru/TensorArrayV2_2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
gru/TensorArrayV2_2Ћ
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2=
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeР
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
:€€€€€€€€€ 2
gru/zeros_likeЗ
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter 
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
gru_while_body_10904* 
condR
gru_while_cond_10903*M
output_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : *
parallel_iterations 2
	gru/whileљ
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    26
4gru/TensorArrayV2Stack/TensorListStack/element_shapeБ
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStackЙ
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
gru/strided_slice_3/stackД
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1Д
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2≤
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
gru/strided_slice_3Б
gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_2/permЊ
gru/transpose_2	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
gru/transpose_2n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtimeЂ
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
output/Tensordot/ShapeВ
output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/GatherV2/axisф
output/Tensordot/GatherV2GatherV2output/Tensordot/Shape:output:0output/Tensordot/free:output:0'output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2Ж
 output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 output/Tensordot/GatherV2_1/axisъ
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
output/Tensordot/ConstЬ
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
output/Tensordot/Const_1§
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
output/Tensordot/concat/axis”
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat®
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stackї
output/Tensordot/transpose	Transposegru/transpose_2:y:0 output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
output/Tensordot/transposeї
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
output/Tensordot/ReshapeЇ
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/Tensordot/MatMul~
output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/Const_2В
output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/concat_1/axisа
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1µ
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
output/Tensordot°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpђ
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
output/BiasAdd÷
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulй
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЮ
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2
	gru/while	gru/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2B
output/Tensordot/ReadVariableOpoutput/Tensordot/ReadVariableOp:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
р
†
while_cond_9926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9926___redundant_placeholder02
.while_while_cond_9926___redundant_placeholder12
.while_while_cond_9926___redundant_placeholder22
.while_while_cond_9926___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
О'
µ
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10636

inputs
	gru_10611:`
	gru_10613:	ђ`
	gru_10615: `
output_10618: 
output_10620:
identityИҐgru/StatefulPartitionedCallҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐoutput/StatefulPartitionedCallв
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_masking_layer_call_and_return_conditional_losses_97952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_10611	gru_10613	gru_10615*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_105752
gru/StatefulPartitionedCallЈ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_10618output_10620*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_101292 
output/StatefulPartitionedCallє
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10613*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulћ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10615*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
™

Ћ
gru_while_cond_10903$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_10903___redundant_placeholder0;
7gru_while_gru_while_cond_10903___redundant_placeholder1;
7gru_while_gru_while_cond_10903___redundant_placeholder2;
7gru_while_gru_while_cond_10903___redundant_placeholder3;
7gru_while_gru_while_cond_10903___redundant_placeholder4
gru_while_identity
Д
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
D: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
шg
”
!__inference__traced_restore_13585
file_prefix0
assignvariableop_output_kernel: ,
assignvariableop_1_output_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: 9
&assignvariableop_7_gru_gru_cell_kernel:	ђ`B
0assignvariableop_8_gru_gru_cell_recurrent_kernel: `6
$assignvariableop_9_gru_gru_cell_bias:`#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: :
(assignvariableop_14_adam_output_kernel_m: 4
&assignvariableop_15_adam_output_bias_m:A
.assignvariableop_16_adam_gru_gru_cell_kernel_m:	ђ`J
8assignvariableop_17_adam_gru_gru_cell_recurrent_kernel_m: `>
,assignvariableop_18_adam_gru_gru_cell_bias_m:`:
(assignvariableop_19_adam_output_kernel_v: 4
&assignvariableop_20_adam_output_bias_v:A
.assignvariableop_21_adam_gru_gru_cell_kernel_v:	ђ`J
8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_v: `>
,assignvariableop_23_adam_gru_gru_cell_bias_v:`
identity_25ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ѓ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ї
value∞B≠B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesј
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices®
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

IdentityЭ
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

Identity_2°
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

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6™
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
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
Identity_10°
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11°
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
Identity_14∞
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_output_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ѓ
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_output_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ґ
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_gru_gru_cell_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ј
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_gru_gru_cell_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18і
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_gru_gru_cell_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19∞
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_output_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_output_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ґ
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_gru_gru_cell_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ј
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23і
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_gru_gru_cell_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpо
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24б
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
ф
C
'__inference_masking_layer_call_fn_11603

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_masking_layer_call_and_return_conditional_losses_97952
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€€€€€€€€€€ђ:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
х
•
while_cond_10362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10362___redundant_placeholder03
/while_while_cond_10362___redundant_placeholder13
/while_while_cond_10362___redundant_placeholder23
/while_while_cond_10362___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
џП
Ђ
"GRU_classifier_gru_while_body_8756B
>gru_classifier_gru_while_gru_classifier_gru_while_loop_counterH
Dgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations(
$gru_classifier_gru_while_placeholder*
&gru_classifier_gru_while_placeholder_1*
&gru_classifier_gru_while_placeholder_2*
&gru_classifier_gru_while_placeholder_3A
=gru_classifier_gru_while_gru_classifier_gru_strided_slice_1_0}
ygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0Б
}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0M
;gru_classifier_gru_while_gru_cell_readvariableop_resource_0:`P
=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0:	ђ`O
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
;gru_classifier_gru_while_gru_cell_readvariableop_1_resource:	ђ`M
;gru_classifier_gru_while_gru_cell_readvariableop_4_resource: `ИҐ0GRU_classifier/gru/while/gru_cell/ReadVariableOpҐ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1Ґ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2Ґ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3Ґ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4Ґ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5Ґ2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6й
JGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  2L
JGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape∆
<GRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0$gru_classifier_gru_while_placeholderSGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02>
<GRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItemн
LGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2N
LGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeѕ
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0$gru_classifier_gru_while_placeholderUGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
2@
>GRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItemў
1GRU_classifier/gru/while/gru_cell/ones_like/ShapeShapeCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1GRU_classifier/gru/while/gru_cell/ones_like/ShapeЂ
1GRU_classifier/gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1GRU_classifier/gru/while/gru_cell/ones_like/ConstН
+GRU_classifier/gru/while/gru_cell/ones_likeFill:GRU_classifier/gru/while/gru_cell/ones_like/Shape:output:0:GRU_classifier/gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2-
+GRU_classifier/gru/while/gru_cell/ones_likeј
3GRU_classifier/gru/while/gru_cell/ones_like_1/ShapeShape&gru_classifier_gru_while_placeholder_3*
T0*
_output_shapes
:25
3GRU_classifier/gru/while/gru_cell/ones_like_1/Shapeѓ
3GRU_classifier/gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?25
3GRU_classifier/gru/while/gru_cell/ones_like_1/ConstФ
-GRU_classifier/gru/while/gru_cell/ones_like_1Fill<GRU_classifier/gru/while/gru_cell/ones_like_1/Shape:output:0<GRU_classifier/gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-GRU_classifier/gru/while/gru_cell/ones_like_1а
0GRU_classifier/gru/while/gru_cell/ReadVariableOpReadVariableOp;gru_classifier_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype022
0GRU_classifier/gru/while/gru_cell/ReadVariableOp–
)GRU_classifier/gru/while/gru_cell/unstackUnpack8GRU_classifier/gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2+
)GRU_classifier/gru/while/gru_cell/unstackГ
%GRU_classifier/gru/while/gru_cell/mulMulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2'
%GRU_classifier/gru/while/gru_cell/mulЗ
'GRU_classifier/gru/while/gru_cell/mul_1MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2)
'GRU_classifier/gru/while/gru_cell/mul_1З
'GRU_classifier/gru/while/gru_cell/mul_2MulCGRU_classifier/gru/while/TensorArrayV2Read/TensorListGetItem:item:04GRU_classifier/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2)
'GRU_classifier/gru/while/gru_cell/mul_2з
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_1њ
5GRU_classifier/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5GRU_classifier/gru/while/gru_cell/strided_slice/stack√
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_1√
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7GRU_classifier/gru/while/gru_cell/strided_slice/stack_2Ћ
/GRU_classifier/gru/while/gru_cell/strided_sliceStridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_1:value:0>GRU_classifier/gru/while/gru_cell/strided_slice/stack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice/stack_1:output:0@GRU_classifier/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask21
/GRU_classifier/gru/while/gru_cell/strided_sliceх
(GRU_classifier/gru/while/gru_cell/MatMulMatMul)GRU_classifier/gru/while/gru_cell/mul:z:08GRU_classifier/gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2*
(GRU_classifier/gru/while/gru_cell/MatMulз
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_2√
7GRU_classifier/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_1/stack«
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1«
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2’
1GRU_classifier/gru/while/gru_cell/strided_slice_1StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_2:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_1/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_1/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_1э
*GRU_classifier/gru/while/gru_cell/MatMul_1MatMul+GRU_classifier/gru/while/gru_cell/mul_1:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_1з
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_3√
7GRU_classifier/gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7GRU_classifier/gru/while/gru_cell/strided_slice_2/stack«
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1«
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2’
1GRU_classifier/gru/while/gru_cell/strided_slice_2StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_3:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_2/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_2/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_2э
*GRU_classifier/gru/while/gru_cell/MatMul_2MatMul+GRU_classifier/gru/while/gru_cell/mul_2:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_2Љ
7GRU_classifier/gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_3/stackј
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1ј
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2Є
1GRU_classifier/gru/while/gru_cell/strided_slice_3StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_3/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_3/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_3Г
)GRU_classifier/gru/while/gru_cell/BiasAddBiasAdd2GRU_classifier/gru/while/gru_cell/MatMul:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)GRU_classifier/gru/while/gru_cell/BiasAddЉ
7GRU_classifier/gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_4/stackј
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1ј
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2¶
1GRU_classifier/gru/while/gru_cell/strided_slice_4StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_4/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_4/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 23
1GRU_classifier/gru/while/gru_cell/strided_slice_4Й
+GRU_classifier/gru/while/gru_cell/BiasAdd_1BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_1:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_1Љ
7GRU_classifier/gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@29
7GRU_classifier/gru/while/gru_cell/strided_slice_5/stackј
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1ј
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2ґ
1GRU_classifier/gru/while/gru_cell/strided_slice_5StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:0@GRU_classifier/gru/while/gru_cell/strided_slice_5/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_5/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_5Й
+GRU_classifier/gru/while/gru_cell/BiasAdd_2BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_2:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_2л
'GRU_classifier/gru/while/gru_cell/mul_3Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/mul_3л
'GRU_classifier/gru/while/gru_cell/mul_4Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/mul_4л
'GRU_classifier/gru/while/gru_cell/mul_5Mul&gru_classifier_gru_while_placeholder_36GRU_classifier/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/mul_5ж
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_4√
7GRU_classifier/gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_6/stack«
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1«
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2‘
1GRU_classifier/gru/while/gru_cell/strided_slice_6StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_4:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_6/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_6/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_6э
*GRU_classifier/gru/while/gru_cell/MatMul_3MatMul+GRU_classifier/gru/while/gru_cell/mul_3:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_3ж
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_5√
7GRU_classifier/gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7GRU_classifier/gru/while/gru_cell/strided_slice_7/stack«
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2;
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1«
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9GRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2‘
1GRU_classifier/gru/while/gru_cell/strided_slice_7StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_5:value:0@GRU_classifier/gru/while/gru_cell/strided_slice_7/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_7/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_7э
*GRU_classifier/gru/while/gru_cell/MatMul_4MatMul+GRU_classifier/gru/while/gru_cell/mul_4:z:0:GRU_classifier/gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_4Љ
7GRU_classifier/gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_8/stackј
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1ј
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2Є
1GRU_classifier/gru/while/gru_cell/strided_slice_8StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1@GRU_classifier/gru/while/gru_cell/strided_slice_8/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_8/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask23
1GRU_classifier/gru/while/gru_cell/strided_slice_8Й
+GRU_classifier/gru/while/gru_cell/BiasAdd_3BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_3:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_3Љ
7GRU_classifier/gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7GRU_classifier/gru/while/gru_cell/strided_slice_9/stackј
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2;
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1ј
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9GRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2¶
1GRU_classifier/gru/while/gru_cell/strided_slice_9StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1@GRU_classifier/gru/while/gru_cell/strided_slice_9/stack:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_9/stack_1:output:0BGRU_classifier/gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 23
1GRU_classifier/gru/while/gru_cell/strided_slice_9Й
+GRU_classifier/gru/while/gru_cell/BiasAdd_4BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_4:product:0:GRU_classifier/gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_4у
%GRU_classifier/gru/while/gru_cell/addAddV22GRU_classifier/gru/while/gru_cell/BiasAdd:output:04GRU_classifier/gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/while/gru_cell/addЊ
)GRU_classifier/gru/while/gru_cell/SigmoidSigmoid)GRU_classifier/gru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)GRU_classifier/gru/while/gru_cell/Sigmoidщ
'GRU_classifier/gru/while/gru_cell/add_1AddV24GRU_classifier/gru/while/gru_cell/BiasAdd_1:output:04GRU_classifier/gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/add_1ƒ
+GRU_classifier/gru/while/gru_cell/Sigmoid_1Sigmoid+GRU_classifier/gru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+GRU_classifier/gru/while/gru_cell/Sigmoid_1ж
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6ReadVariableOp=gru_classifier_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype024
2GRU_classifier/gru/while/gru_cell/ReadVariableOp_6≈
8GRU_classifier/gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2:
8GRU_classifier/gru/while/gru_cell/strided_slice_10/stack…
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1…
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:GRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2ў
2GRU_classifier/gru/while/gru_cell/strided_slice_10StridedSlice:GRU_classifier/gru/while/gru_cell/ReadVariableOp_6:value:0AGRU_classifier/gru/while/gru_cell/strided_slice_10/stack:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_10/stack_1:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask24
2GRU_classifier/gru/while/gru_cell/strided_slice_10ю
*GRU_classifier/gru/while/gru_cell/MatMul_5MatMul+GRU_classifier/gru/while/gru_cell/mul_5:z:0;GRU_classifier/gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*GRU_classifier/gru/while/gru_cell/MatMul_5Њ
8GRU_classifier/gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2:
8GRU_classifier/gru/while/gru_cell/strided_slice_11/stack¬
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1¬
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:GRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2ї
2GRU_classifier/gru/while/gru_cell/strided_slice_11StridedSlice2GRU_classifier/gru/while/gru_cell/unstack:output:1AGRU_classifier/gru/while/gru_cell/strided_slice_11/stack:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_11/stack_1:output:0CGRU_classifier/gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask24
2GRU_classifier/gru/while/gru_cell/strided_slice_11К
+GRU_classifier/gru/while/gru_cell/BiasAdd_5BiasAdd4GRU_classifier/gru/while/gru_cell/MatMul_5:product:0;GRU_classifier/gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+GRU_classifier/gru/while/gru_cell/BiasAdd_5т
'GRU_classifier/gru/while/gru_cell/mul_6Mul/GRU_classifier/gru/while/gru_cell/Sigmoid_1:y:04GRU_classifier/gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/mul_6р
'GRU_classifier/gru/while/gru_cell/add_2AddV24GRU_classifier/gru/while/gru_cell/BiasAdd_2:output:0+GRU_classifier/gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/add_2Ј
&GRU_classifier/gru/while/gru_cell/TanhTanh+GRU_classifier/gru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&GRU_classifier/gru/while/gru_cell/Tanhв
'GRU_classifier/gru/while/gru_cell/mul_7Mul-GRU_classifier/gru/while/gru_cell/Sigmoid:y:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/mul_7Ч
'GRU_classifier/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'GRU_classifier/gru/while/gru_cell/sub/xи
%GRU_classifier/gru/while/gru_cell/subSub0GRU_classifier/gru/while/gru_cell/sub/x:output:0-GRU_classifier/gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%GRU_classifier/gru/while/gru_cell/subв
'GRU_classifier/gru/while/gru_cell/mul_8Mul)GRU_classifier/gru/while/gru_cell/sub:z:0*GRU_classifier/gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/mul_8з
'GRU_classifier/gru/while/gru_cell/add_3AddV2+GRU_classifier/gru/while/gru_cell/mul_7:z:0+GRU_classifier/gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'GRU_classifier/gru/while/gru_cell/add_3£
'GRU_classifier/gru/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'GRU_classifier/gru/while/Tile/multiplesс
GRU_classifier/gru/while/TileTileEGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem:item:00GRU_classifier/gru/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
GRU_classifier/gru/while/TileБ
!GRU_classifier/gru/while/SelectV2SelectV2&GRU_classifier/gru/while/Tile:output:0+GRU_classifier/gru/while/gru_cell/add_3:z:0&gru_classifier_gru_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2#
!GRU_classifier/gru/while/SelectV2І
)GRU_classifier/gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)GRU_classifier/gru/while/Tile_1/multiplesч
GRU_classifier/gru/while/Tile_1TileEGRU_classifier/gru/while/TensorArrayV2Read_1/TensorListGetItem:item:02GRU_classifier/gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2!
GRU_classifier/gru/while/Tile_1З
#GRU_classifier/gru/while/SelectV2_1SelectV2(GRU_classifier/gru/while/Tile_1:output:0+GRU_classifier/gru/while/gru_cell/add_3:z:0&gru_classifier_gru_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#GRU_classifier/gru/while/SelectV2_1Ї
=GRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&gru_classifier_gru_while_placeholder_1$gru_classifier_gru_while_placeholder*GRU_classifier/gru/while/SelectV2:output:0*
_output_shapes
: *
element_dtype02?
=GRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItemВ
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
GRU_classifier/gru/while/addЖ
 GRU_classifier/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 GRU_classifier/gru/while/add_1/y’
GRU_classifier/gru/while/add_1AddV2>gru_classifier_gru_while_gru_classifier_gru_while_loop_counter)GRU_classifier/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
GRU_classifier/gru/while/add_1И
!GRU_classifier/gru/while/IdentityIdentity"GRU_classifier/gru/while/add_1:z:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2#
!GRU_classifier/gru/while/IdentityЃ
#GRU_classifier/gru/while/Identity_1IdentityDgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations1^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_1К
#GRU_classifier/gru/while/Identity_2Identity GRU_classifier/gru/while/add:z:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_2Ј
#GRU_classifier/gru/while/Identity_3IdentityMGRU_classifier/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2%
#GRU_classifier/gru/while/Identity_3•
#GRU_classifier/gru/while/Identity_4Identity*GRU_classifier/gru/while/SelectV2:output:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#GRU_classifier/gru/while/Identity_4І
#GRU_classifier/gru/while/Identity_5Identity,GRU_classifier/gru/while/SelectV2_1:output:01^GRU_classifier/gru/while/gru_cell/ReadVariableOp3^GRU_classifier/gru/while/gru_cell/ReadVariableOp_13^GRU_classifier/gru/while/gru_cell/ReadVariableOp_23^GRU_classifier/gru/while/gru_cell/ReadVariableOp_33^GRU_classifier/gru/while/gru_cell/ReadVariableOp_43^GRU_classifier/gru/while/gru_cell/ReadVariableOp_53^GRU_classifier/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2%
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
#gru_classifier_gru_while_identity_5,GRU_classifier/gru/while/Identity_5:output:0"ь
{gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor}gru_classifier_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_1_tensorlistfromtensor_0"ф
wgru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensorygru_classifier_gru_while_tensorarrayv2read_tensorlistgetitem_gru_classifier_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : 2d
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЎО
¬
>__inference_gru_layer_call_and_return_conditional_losses_10575

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	ђ`4
"gru_cell_readvariableop_4_resource: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐgru_cell/ReadVariableOpҐgru_cell/ReadVariableOp_1Ґgru_cell/ReadVariableOp_2Ґgru_cell/ReadVariableOp_3Ґgru_cell/ReadVariableOp_4Ґgru_cell/ReadVariableOp_5Ґgru_cell/ReadVariableOp_6ҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
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
 *  А?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout/Const§
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeл
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2иЦt2/
-gru_cell/dropout/random_uniform/RandomUniformЗ
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2!
gru_cell/dropout/GreaterEqual/yг
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/GreaterEqualЫ
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/CastЯ
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_1/Const™
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeт
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2БЕ«21
/gru_cell/dropout_1/random_uniform/RandomUniformЛ
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_1/GreaterEqual/yл
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
gru_cell/dropout_1/GreaterEqual°
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/CastІ
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_2/Const™
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeт
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ј‘”21
/gru_cell/dropout_2/random_uniform/RandomUniformЛ
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_2/GreaterEqual/yл
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
gru_cell/dropout_2/GreaterEqual°
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_2/CastІ
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
gru_cell/ones_like_1/Const∞
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_3/ConstЂ
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/MulБ
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeр
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ыЬu21
/gru_cell/dropout_3/random_uniform/RandomUniformЛ
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_3/GreaterEqual/yк
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_3/GreaterEqual†
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/Cast¶
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_4/ConstЂ
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/MulБ
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeс
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ёо≈21
/gru_cell/dropout_4/random_uniform/RandomUniformЛ
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_4/GreaterEqual/yк
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_4/GreaterEqual†
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/Cast¶
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_5/ConstЂ
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/MulБ
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeс
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ў§Б21
/gru_cell/dropout_5/random_uniform/RandomUniformЛ
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_5/GreaterEqual/yк
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_5/GreaterEqual†
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/Cast¶
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/Mul_1У
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOpЕ
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstackМ
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mulТ
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_1Т
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_2Ъ
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_1Н
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stackС
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1С
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
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_sliceС
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMulЪ
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_2С
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stackХ
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1Х
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2њ
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_1Щ
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_1Ъ
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_3С
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stackХ
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1Х
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2њ
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_2Щ
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_2К
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stackО
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1О
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2Ґ
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3Я
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAddК
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stackО
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1О
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2Р
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4•
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_1К
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stackО
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1О
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2†
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5•
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_2З
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_3З
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_4З
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_5Щ
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4С
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stackХ
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1Х
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2Њ
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6Щ
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_3Щ
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5С
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stackХ
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1Х
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2Њ
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7Щ
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_4К
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stackО
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1О
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2Ґ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8•
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_3К
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stackО
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1О
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2Р
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9•
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_4П
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/SigmoidХ
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Sigmoid_1Щ
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6У
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stackЧ
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1Ч
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2√
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10Ъ
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_5М
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stackР
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1Р
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2•
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¶
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_5О
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_6М
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru_cell/sub/xД
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_8Г
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10363*
condR
while_cond_10362*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime“
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulе
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulі
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
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
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
в
т
.__inference_GRU_classifier_layer_call_fn_10664	
input
unknown:`
	unknown_0:	ђ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8В *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_106362
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_nameinput
Е
П
"GRU_classifier_gru_while_cond_8755B
>gru_classifier_gru_while_gru_classifier_gru_while_loop_counterH
Dgru_classifier_gru_while_gru_classifier_gru_while_maximum_iterations(
$gru_classifier_gru_while_placeholder*
&gru_classifier_gru_while_placeholder_1*
&gru_classifier_gru_while_placeholder_2*
&gru_classifier_gru_while_placeholder_3D
@gru_classifier_gru_while_less_gru_classifier_gru_strided_slice_1X
Tgru_classifier_gru_while_gru_classifier_gru_while_cond_8755___redundant_placeholder0X
Tgru_classifier_gru_while_gru_classifier_gru_while_cond_8755___redundant_placeholder1X
Tgru_classifier_gru_while_gru_classifier_gru_while_cond_8755___redundant_placeholder2X
Tgru_classifier_gru_while_gru_classifier_gru_while_cond_8755___redundant_placeholder3X
Tgru_classifier_gru_while_gru_classifier_gru_while_cond_8755___redundant_placeholder4%
!gru_classifier_gru_while_identity
ѕ
GRU_classifier/gru/while/LessLess$gru_classifier_gru_while_placeholder@gru_classifier_gru_while_less_gru_classifier_gru_strided_slice_1*
T0*
_output_shapes
: 2
GRU_classifier/gru/while/LessЦ
!GRU_classifier/gru/while/IdentityIdentity!GRU_classifier/gru/while/Less:z:0*
T0
*
_output_shapes
: 2#
!GRU_classifier/gru/while/Identity"O
!gru_classifier_gru_while_identity*GRU_classifier/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
х
•
while_cond_11745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_11745___redundant_placeholder03
/while_while_cond_11745___redundant_placeholder13
/while_while_cond_11745___redundant_placeholder23
/while_while_cond_11745___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
√Е
Г
C__inference_gru_cell_layer_call_and_return_conditional_losses_13196

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	ђ`+
readvariableop_4_resource: `
identity

identity_1ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3ҐReadVariableOp_4ҐReadVariableOp_5ҐReadVariableOp_6Ґ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
 *  А?2
ones_like/ConstЕ
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
ones_like_1/ConstМ
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ђ2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
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
strided_slice/stack_2€
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Й
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2Й
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
strided_slice_3/stack_2м
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
:€€€€€€€€€ 2	
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
strided_slice_4/stack_2Џ
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4Б
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_5/stack_2к
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5Б
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2g
mul_3Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_8/stack_2м
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8Б
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_9/stack_2Џ
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9Б
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6Б
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2Н
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
:€€€€€€€€€ 2

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
strided_slice_11/stack_2п
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11В
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_3…
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul№
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЏ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
≠
з
#__inference_signature_wrapper_10757	
input
unknown:`
	unknown_0:	ђ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8В *(
f#R!
__inference__wrapped_model_89492
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_nameinput
≈ў
≠
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_11557

inputs6
$gru_gru_cell_readvariableop_resource:`9
&gru_gru_cell_readvariableop_1_resource:	ђ`8
&gru_gru_cell_readvariableop_4_resource: `:
(output_tensordot_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identityИҐgru/gru_cell/ReadVariableOpҐgru/gru_cell/ReadVariableOp_1Ґgru/gru_cell/ReadVariableOp_2Ґgru/gru_cell/ReadVariableOp_3Ґgru/gru_cell/ReadVariableOp_4Ґgru/gru_cell/ReadVariableOp_5Ґgru/gru_cell/ReadVariableOp_6Ґ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ	gru/whileҐoutput/BiasAdd/ReadVariableOpҐoutput/Tensordot/ReadVariableOpm
masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking/NotEqual/yХ
masking/NotEqualNotEqualinputsmasking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
masking/NotEqualЙ
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
masking/Any/reduction_indices¶
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
	keep_dims(2
masking/AnyИ
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
masking/mulЮ
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
squeeze_dims

€€€€€€€€€2
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
gru/strided_slice/stackА
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1А
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ъ
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
B :и2
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
gru/zeros/packed/1У
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
gru/zeros/ConstЕ
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/permЩ
gru/transpose	Transposemasking/mul:z:0gru/transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1А
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stackД
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1Д
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2Ж
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
€€€€€€€€€2
gru/ExpandDims/dim§
gru/ExpandDims
ExpandDimsmasking/Squeeze:output:0gru/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
gru/ExpandDimsБ
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm¶
gru/transpose_1	Transposegru/ExpandDims:output:0gru/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
gru/transpose_1Н
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
gru/TensorArrayV2/element_shape¬
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2«
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensorА
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stackД
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1Д
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2Х
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_mask2
gru/strided_slice_2И
gru/gru_cell/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/ShapeБ
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru/gru_cell/ones_like/Constє
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/ones_like}
gru/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru/gru_cell/dropout/Constі
gru/gru_cell/dropout/MulMulgru/gru_cell/ones_like:output:0#gru/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout/MulЗ
gru/gru_cell/dropout/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout/Shapeч
1gru/gru_cell/dropout/random_uniform/RandomUniformRandomUniform#gru/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ў≠N23
1gru/gru_cell/dropout/random_uniform/RandomUniformП
#gru/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2%
#gru/gru_cell/dropout/GreaterEqual/yу
!gru/gru_cell/dropout/GreaterEqualGreaterEqual:gru/gru_cell/dropout/random_uniform/RandomUniform:output:0,gru/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2#
!gru/gru_cell/dropout/GreaterEqualІ
gru/gru_cell/dropout/CastCast%gru/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout/Castѓ
gru/gru_cell/dropout/Mul_1Mulgru/gru_cell/dropout/Mul:z:0gru/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout/Mul_1Б
gru/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru/gru_cell/dropout_1/ConstЇ
gru/gru_cell/dropout_1/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout_1/MulЛ
gru/gru_cell/dropout_1/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_1/Shapeю
3gru/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2Юхд25
3gru/gru_cell/dropout_1/random_uniform/RandomUniformУ
%gru/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%gru/gru_cell/dropout_1/GreaterEqual/yы
#gru/gru_cell/dropout_1/GreaterEqualGreaterEqual<gru/gru_cell/dropout_1/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2%
#gru/gru_cell/dropout_1/GreaterEqual≠
gru/gru_cell/dropout_1/CastCast'gru/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout_1/CastЈ
gru/gru_cell/dropout_1/Mul_1Mulgru/gru_cell/dropout_1/Mul:z:0gru/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout_1/Mul_1Б
gru/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru/gru_cell/dropout_2/ConstЇ
gru/gru_cell/dropout_2/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout_2/MulЛ
gru/gru_cell/dropout_2/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_2/Shapeю
3gru/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2іЬб25
3gru/gru_cell/dropout_2/random_uniform/RandomUniformУ
%gru/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%gru/gru_cell/dropout_2/GreaterEqual/yы
#gru/gru_cell/dropout_2/GreaterEqualGreaterEqual<gru/gru_cell/dropout_2/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2%
#gru/gru_cell/dropout_2/GreaterEqual≠
gru/gru_cell/dropout_2/CastCast'gru/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout_2/CastЈ
gru/gru_cell/dropout_2/Mul_1Mulgru/gru_cell/dropout_2/Mul:z:0gru/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/dropout_2/Mul_1В
gru/gru_cell/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2 
gru/gru_cell/ones_like_1/ShapeЕ
gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
gru/gru_cell/ones_like_1/Constј
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/ones_like_1Б
gru/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru/gru_cell/dropout_3/Constї
gru/gru_cell/dropout_3/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_3/MulН
gru/gru_cell/dropout_3/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_3/Shapeэ
3gru/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2эЭЋ25
3gru/gru_cell/dropout_3/random_uniform/RandomUniformУ
%gru/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%gru/gru_cell/dropout_3/GreaterEqual/yъ
#gru/gru_cell/dropout_3/GreaterEqualGreaterEqual<gru/gru_cell/dropout_3/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#gru/gru_cell/dropout_3/GreaterEqualђ
gru/gru_cell/dropout_3/CastCast'gru/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_3/Castґ
gru/gru_cell/dropout_3/Mul_1Mulgru/gru_cell/dropout_3/Mul:z:0gru/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_3/Mul_1Б
gru/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru/gru_cell/dropout_4/Constї
gru/gru_cell/dropout_4/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_4/MulН
gru/gru_cell/dropout_4/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_4/Shapeэ
3gru/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ќэ≤25
3gru/gru_cell/dropout_4/random_uniform/RandomUniformУ
%gru/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%gru/gru_cell/dropout_4/GreaterEqual/yъ
#gru/gru_cell/dropout_4/GreaterEqualGreaterEqual<gru/gru_cell/dropout_4/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#gru/gru_cell/dropout_4/GreaterEqualђ
gru/gru_cell/dropout_4/CastCast'gru/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_4/Castґ
gru/gru_cell/dropout_4/Mul_1Mulgru/gru_cell/dropout_4/Mul:z:0gru/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_4/Mul_1Б
gru/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru/gru_cell/dropout_5/Constї
gru/gru_cell/dropout_5/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_5/MulН
gru/gru_cell/dropout_5/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_5/Shapeэ
3gru/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2„Ќњ25
3gru/gru_cell/dropout_5/random_uniform/RandomUniformУ
%gru/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2'
%gru/gru_cell/dropout_5/GreaterEqual/yъ
#gru/gru_cell/dropout_5/GreaterEqualGreaterEqual<gru/gru_cell/dropout_5/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#gru/gru_cell/dropout_5/GreaterEqualђ
gru/gru_cell/dropout_5/CastCast'gru/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_5/Castґ
gru/gru_cell/dropout_5/Mul_1Mulgru/gru_cell/dropout_5/Mul:z:0gru/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/dropout_5/Mul_1Я
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell/ReadVariableOpС
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell/unstackЬ
gru/gru_cell/mulMulgru/strided_slice_2:output:0gru/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/mulҐ
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/mul_1Ґ
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/gru_cell/mul_2¶
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru/gru_cell/ReadVariableOp_1Х
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stackЩ
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice/stack_1Щ
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2Ќ
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice°
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul¶
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru/gru_cell/ReadVariableOp_2Щ
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_1/stackЭ
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_1/stack_1Э
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2„
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1©
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul_1¶
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru/gru_cell/ReadVariableOp_3Щ
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"gru/gru_cell/strided_slice_2/stackЭ
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1Э
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2„
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2©
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul_2Т
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stackЦ
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_3/stack_1Ц
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2Ї
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru/gru_cell/strided_slice_3ѓ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAddТ
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_4/stackЦ
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_4/stack_1Ц
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2®
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_4µ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_1Т
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2$
"gru/gru_cell/strided_slice_5/stackЦ
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1Ц
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2Є
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
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_2Ч
gru/gru_cell/mul_3Mulgru/zeros:output:0 gru/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_3Ч
gru/gru_cell/mul_4Mulgru/zeros:output:0 gru/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_4Ч
gru/gru_cell/mul_5Mulgru/zeros:output:0 gru/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_5•
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_4Щ
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stackЭ
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_6/stack_1Э
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2÷
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
:€€€€€€€€€ 2
gru/gru_cell/MatMul_3•
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_5Щ
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_7/stackЭ
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$gru/gru_cell/strided_slice_7/stack_1Э
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2÷
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
:€€€€€€€€€ 2
gru/gru_cell/MatMul_4Т
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stackЦ
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_8/stack_1Ц
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2Ї
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
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_3Т
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_9/stackЦ
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2&
$gru/gru_cell/strided_slice_9/stack_1Ц
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2®
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru/gru_cell/strided_slice_9µ
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_4Я
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/Sigmoid•
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add_1Е
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/Sigmoid_1•
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru/gru_cell/ReadVariableOp_6Ы
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2%
#gru/gru_cell/strided_slice_10/stackЯ
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1Я
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2џ
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10™
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/MatMul_5Ф
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2%
#gru/gru_cell/strided_slice_11/stackШ
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1Ш
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2љ
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru/gru_cell/strided_slice_11ґ
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/BiasAdd_5Ю
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_6Ь
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/TanhП
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_7m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru/gru_cell/sub/xФ
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/subО
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/mul_8У
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/gru_cell/add_3Ч
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2#
!gru/TensorArrayV2_1/element_shape»
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

gru/timeС
!gru/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2#
!gru/TensorArrayV2_2/element_shape»
gru/TensorArrayV2_2TensorListReserve*gru/TensorArrayV2_2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
gru/TensorArrayV2_2Ћ
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2=
;gru/TensorArrayUnstack_1/TensorListFromTensor/element_shapeР
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
:€€€€€€€€€ 2
gru/zeros_likeЗ
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter 
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros_like:y:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=gru/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
gru_while_body_11304* 
condR
gru_while_cond_11303*M
output_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : *
parallel_iterations 2
	gru/whileљ
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    26
4gru/TensorArrayV2Stack/TensorListStack/element_shapeБ
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStackЙ
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
gru/strided_slice_3/stackД
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1Д
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2≤
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
gru/strided_slice_3Б
gru/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_2/permЊ
gru/transpose_2	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
gru/transpose_2n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtimeЂ
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
output/Tensordot/ShapeВ
output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/GatherV2/axisф
output/Tensordot/GatherV2GatherV2output/Tensordot/Shape:output:0output/Tensordot/free:output:0'output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
output/Tensordot/GatherV2Ж
 output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 output/Tensordot/GatherV2_1/axisъ
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
output/Tensordot/ConstЬ
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
output/Tensordot/Const_1§
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
output/Tensordot/concat/axis”
output/Tensordot/concatConcatV2output/Tensordot/free:output:0output/Tensordot/axes:output:0%output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat®
output/Tensordot/stackPackoutput/Tensordot/Prod:output:0 output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/stackї
output/Tensordot/transpose	Transposegru/transpose_2:y:0 output/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
output/Tensordot/transposeї
output/Tensordot/ReshapeReshapeoutput/Tensordot/transpose:y:0output/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
output/Tensordot/ReshapeЇ
output/Tensordot/MatMulMatMul!output/Tensordot/Reshape:output:0'output/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/Tensordot/MatMul~
output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
output/Tensordot/Const_2В
output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
output/Tensordot/concat_1/axisа
output/Tensordot/concat_1ConcatV2"output/Tensordot/GatherV2:output:0!output/Tensordot/Const_2:output:0'output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
output/Tensordot/concat_1µ
output/TensordotReshape!output/Tensordot/MatMul:product:0"output/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
output/Tensordot°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpђ
output/BiasAddBiasAddoutput/Tensordot:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
output/BiasAdd÷
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulй
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЮ
IdentityIdentityoutput/BiasAdd:output:0^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
^gru/while^output/BiasAdd/ReadVariableOp ^output/Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2
	gru/while	gru/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2B
output/Tensordot/ReadVariableOpoutput/Tensordot/ReadVariableOp:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
л	
^
B__inference_masking_layer_call_and_return_conditional_losses_11598

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
!:€€€€€€€€€€€€€€€€€€ђ2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Any/reduction_indicesЖ
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
mulЖ
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
squeeze_dims

€€€€€€€€€2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€€€€€€€€€€ђ:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
Ґќ
ƒ
>__inference_gru_layer_call_and_return_conditional_losses_11910
inputs_02
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	ђ`4
"gru_cell_readvariableop_4_resource: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐgru_cell/ReadVariableOpҐgru_cell/ReadVariableOp_1Ґgru_cell/ReadVariableOp_2Ґgru_cell/ReadVariableOp_3Ґgru_cell/ReadVariableOp_4Ґgru_cell/ReadVariableOp_5Ґgru_cell/ReadVariableOp_6ҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЖ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
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
 *  А?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
gru_cell/ones_like_1/Const∞
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/ones_like_1У
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOpЕ
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstackН
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mulС
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_1С
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_2Ъ
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_1Н
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stackС
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1С
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
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_sliceС
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMulЪ
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_2С
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stackХ
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1Х
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2њ
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_1Щ
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_1Ъ
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_3С
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stackХ
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1Х
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2њ
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_2Щ
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_2К
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stackО
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1О
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2Ґ
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3Я
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAddК
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stackО
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1О
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2Р
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4•
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_1К
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stackО
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1О
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2†
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5•
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_2И
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_3И
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_4И
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_5Щ
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4С
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stackХ
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1Х
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2Њ
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6Щ
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_3Щ
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5С
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stackХ
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1Х
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2Њ
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7Щ
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_4К
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stackО
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1О
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2Ґ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8•
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_3К
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stackО
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1О
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2Р
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9•
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_4П
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/SigmoidХ
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Sigmoid_1Щ
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6У
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stackЧ
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1Ч
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2√
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10Ъ
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_5М
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stackР
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1Р
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2•
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¶
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_5О
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_6М
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru_cell/sub/xД
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_8Г
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_11746*
condR
while_cond_11745*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime“
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulе
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulі
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
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
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
™≤
ћ
while_body_11746
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
(while_gru_cell_readvariableop_resource_0:`=
*while_gru_cell_readvariableop_1_resource_0:	ђ`<
*while_gru_cell_readvariableop_4_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
&while_gru_cell_readvariableop_resource:`;
(while_gru_cell_readvariableop_1_resource:	ђ`:
(while_gru_cell_readvariableop_4_resource: `ИҐwhile/gru_cell/ReadVariableOpҐwhile/gru_cell/ReadVariableOp_1Ґwhile/gru_cell/ReadVariableOp_2Ґwhile/gru_cell/ReadVariableOp_3Ґwhile/gru_cell/ReadVariableOp_4Ґwhile/gru_cell/ReadVariableOp_5Ґwhile/gru_cell/ReadVariableOp_6√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem†
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/ShapeЕ
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
while/gru_cell/ones_like/ConstЅ
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/ones_likeЗ
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/ShapeЙ
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 while/gru_cell/ones_like_1/Const»
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/ones_like_1І
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOpЧ
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstackЈ
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mulї
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_1ї
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
while/gru_cell/mul_2Ѓ
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_1Щ
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stackЭ
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice/stack_1Э
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2ў
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
while/gru_cell/strided_slice©
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMulЃ
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_2Э
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_1/stack°
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_1/stack_1°
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2г
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1±
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_1Ѓ
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02!
while/gru_cell/ReadVariableOp_3Э
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$while/gru_cell/strided_slice_2/stack°
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1°
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2г
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2±
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_2Ц
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stackЪ
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_3/stack_1Ъ
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2∆
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_3Ј
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAddЦ
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_4/stackЪ
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_4/stack_1Ъ
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2і
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_4љ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_1Ц
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2&
$while/gru_cell/strided_slice_5/stackЪ
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1Ъ
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2ƒ
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
while/gru_cell/strided_slice_5љ
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_2Я
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_3Я
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_4Я
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_5≠
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_4Э
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack°
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_6/stack_1°
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_3≠
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_5Э
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_7/stack°
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2(
&while/gru_cell/strided_slice_7/stack_1°
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2в
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
:€€€€€€€€€ 2
while/gru_cell/MatMul_4Ц
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stackЪ
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_8/stack_1Ъ
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2∆
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
while/gru_cell/strided_slice_8љ
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_3Ц
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_9/stackЪ
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2(
&while/gru_cell/strided_slice_9/stack_1Ъ
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2і
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2 
while/gru_cell/strided_slice_9љ
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_4І
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/addЕ
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid≠
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_1Л
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/Sigmoid_1≠
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02!
while/gru_cell/ReadVariableOp_6Я
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
'while/gru_cell/strided_slice_10/stack_2з
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10≤
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/MatMul_5Ш
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2'
%while/gru_cell/strided_slice_11/stackЬ
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1Ь
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2…
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2!
while/gru_cell/strided_slice_11Њ
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/BiasAdd_5¶
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_6§
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/TanhЦ
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/gru_cell/sub/xЬ
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/subЦ
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/mul_8Ы
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/gru_cell/add_3№
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
while/add_1 
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/IdentityЁ
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1ћ
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3и
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2>
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ЎО
¬
>__inference_gru_layer_call_and_return_conditional_losses_12987

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	ђ`4
"gru_cell_readvariableop_4_resource: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐgru_cell/ReadVariableOpҐgru_cell/ReadVariableOp_1Ґgru_cell/ReadVariableOp_2Ґgru_cell/ReadVariableOp_3Ґgru_cell/ReadVariableOp_4Ґgru_cell/ReadVariableOp_5Ґgru_cell/ReadVariableOp_6ҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
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
 *  А?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout/Const§
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeм
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2н«М2/
-gru_cell/dropout/random_uniform/RandomUniformЗ
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2!
gru_cell/dropout/GreaterEqual/yг
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/GreaterEqualЫ
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/CastЯ
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_1/Const™
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeс
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2жю*21
/gru_cell/dropout_1/random_uniform/RandomUniformЛ
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_1/GreaterEqual/yл
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
gru_cell/dropout_1/GreaterEqual°
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/CastІ
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_2/Const™
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeт
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2Ўь∞21
/gru_cell/dropout_2/random_uniform/RandomUniformЛ
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_2/GreaterEqual/yл
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
gru_cell/dropout_2/GreaterEqual°
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_2/CastІ
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
gru_cell/ones_like_1/Const∞
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_3/ConstЂ
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/MulБ
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeс
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2≈Ђђ21
/gru_cell/dropout_3/random_uniform/RandomUniformЛ
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_3/GreaterEqual/yк
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_3/GreaterEqual†
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/Cast¶
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_4/ConstЂ
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/MulБ
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeс
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2Т∆я21
/gru_cell/dropout_4/random_uniform/RandomUniformЛ
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_4/GreaterEqual/yк
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_4/GreaterEqual†
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/Cast¶
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_5/ConstЂ
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/MulБ
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeр
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2њјw21
/gru_cell/dropout_5/random_uniform/RandomUniformЛ
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_5/GreaterEqual/yк
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_5/GreaterEqual†
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/Cast¶
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/Mul_1У
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOpЕ
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstackМ
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mulТ
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_1Т
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_2Ъ
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_1Н
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stackС
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1С
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
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_sliceС
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMulЪ
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_2С
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stackХ
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1Х
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2њ
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_1Щ
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_1Ъ
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_3С
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stackХ
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1Х
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2њ
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_2Щ
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_2К
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stackО
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1О
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2Ґ
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3Я
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAddК
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stackО
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1О
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2Р
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4•
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_1К
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stackО
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1О
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2†
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5•
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_2З
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_3З
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_4З
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_5Щ
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4С
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stackХ
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1Х
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2Њ
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6Щ
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_3Щ
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5С
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stackХ
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1Х
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2Њ
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7Щ
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_4К
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stackО
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1О
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2Ґ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8•
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_3К
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stackО
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1О
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2Р
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9•
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_4П
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/SigmoidХ
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Sigmoid_1Щ
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6У
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stackЧ
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1Ч
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2√
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10Ъ
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_5М
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stackР
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1Р
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2•
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¶
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_5О
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_6М
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru_cell/sub/xД
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_8Г
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_12775*
condR
while_cond_12774*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime“
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulе
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulі
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
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
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
ш!
О
while_body_9443
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0'
while_gru_cell_9465_0:`(
while_gru_cell_9467_0:	ђ`'
while_gru_cell_9469_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor%
while_gru_cell_9465:`&
while_gru_cell_9467:	ђ`%
while_gru_cell_9469: `ИҐ&while/gru_cell/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9465_0while_gru_cell_9467_0while_gru_cell_9469_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_93762(
&while/gru_cell/StatefulPartitionedCallу
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
while/add_1З
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/IdentityЪ
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1Й
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2ґ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Љ
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4",
while_gru_cell_9465while_gru_cell_9465_0",
while_gru_cell_9467while_gru_cell_9467_0",
while_gru_cell_9469while_gru_cell_9469_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
яО
ƒ
>__inference_gru_layer_call_and_return_conditional_losses_12301
inputs_02
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	ђ`4
"gru_cell_readvariableop_4_resource: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐgru_cell/ReadVariableOpҐgru_cell/ReadVariableOp_1Ґgru_cell/ReadVariableOp_2Ґgru_cell/ReadVariableOp_3Ґgru_cell/ReadVariableOp_4Ґgru_cell/ReadVariableOp_5Ґgru_cell/ReadVariableOp_6ҐwhileF
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЖ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
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
 *  А?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout/Const§
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shapeл
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ІЙf2/
-gru_cell/dropout/random_uniform/RandomUniformЗ
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2!
gru_cell/dropout/GreaterEqual/yг
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/GreaterEqualЫ
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/CastЯ
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_1/Const™
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shapeс
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ы™121
/gru_cell/dropout_1/random_uniform/RandomUniformЛ
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_1/GreaterEqual/yл
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
gru_cell/dropout_1/GreaterEqual°
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/CastІ
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_2/Const™
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shapeт
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ћ …21
/gru_cell/dropout_2/random_uniform/RandomUniformЛ
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_2/GreaterEqual/yл
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2!
gru_cell/dropout_2/GreaterEqual°
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/dropout_2/CastІ
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
gru_cell/ones_like_1/Const∞
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_3/ConstЂ
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/MulБ
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shapeс
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2знЄ21
/gru_cell/dropout_3/random_uniform/RandomUniformЛ
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_3/GreaterEqual/yк
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_3/GreaterEqual†
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/Cast¶
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_4/ConstЂ
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/MulБ
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shapeр
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2Љј521
/gru_cell/dropout_4/random_uniform/RandomUniformЛ
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_4/GreaterEqual/yк
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_4/GreaterEqual†
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/Cast¶
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
gru_cell/dropout_5/ConstЂ
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/MulБ
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shapeс
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2аКљ21
/gru_cell/dropout_5/random_uniform/RandomUniformЛ
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2#
!gru_cell/dropout_5/GreaterEqual/yк
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
gru_cell/dropout_5/GreaterEqual†
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/Cast¶
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/dropout_5/Mul_1У
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOpЕ
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstackМ
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mulТ
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_1Т
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_2Ъ
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_1Н
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stackС
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1С
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
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_sliceС
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMulЪ
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_2С
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stackХ
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1Х
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2њ
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_1Щ
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_1Ъ
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_3С
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stackХ
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1Х
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2њ
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_2Щ
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_2К
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stackО
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1О
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2Ґ
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3Я
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAddК
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stackО
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1О
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2Р
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4•
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_1К
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stackО
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1О
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2†
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5•
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_2З
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_3З
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_4З
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_5Щ
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4С
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stackХ
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1Х
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2Њ
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6Щ
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_3Щ
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5С
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stackХ
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1Х
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2Њ
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7Щ
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_4К
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stackО
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1О
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2Ґ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8•
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_3К
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stackО
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1О
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2Р
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9•
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_4П
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/SigmoidХ
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Sigmoid_1Щ
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6У
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stackЧ
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1Ч
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2√
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10Ъ
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_5М
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stackР
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1Р
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2•
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¶
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_5О
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_6М
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru_cell/sub/xД
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_8Г
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_12089*
condR
while_cond_12088*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime“
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulе
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulі
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
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
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
Л'
і
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10722	
input
	gru_10697:`
	gru_10699:	ђ`
	gru_10701: `
output_10704: 
output_10706:
identityИҐgru/StatefulPartitionedCallҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐoutput/StatefulPartitionedCallб
masking/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_masking_layer_call_and_return_conditional_losses_97952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_10697	gru_10699	gru_10701*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_105752
gru/StatefulPartitionedCallЈ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_10704output_10706*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_101292 
output/StatefulPartitionedCallє
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10699*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulћ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10701*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_nameinput
™

Ћ
gru_while_cond_11303$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2
gru_while_placeholder_3&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_11303___redundant_placeholder0;
7gru_while_gru_while_cond_11303___redundant_placeholder1;
7gru_while_gru_while_cond_11303___redundant_placeholder2;
7gru_while_gru_while_cond_11303___redundant_placeholder3;
7gru_while_gru_while_cond_11303___redundant_placeholder4
gru_while_identity
Д
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
D: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::: 
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ƒ
є
__inference_loss_fn_0_13397Q
>gru_gru_cell_kernel_regularizer_square_readvariableop_resource:	ђ`
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpо
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>gru_gru_cell_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulҐ
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
†8
І

__inference__traced_save_13503
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

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename®
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ї
value∞B≠B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЃ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*Є
_input_shapes¶
£: : :: : : : : :	ђ`: `:`: : : : : ::	ђ`: `:`: ::	ђ`: `:`: 2(
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
:	ђ`:$	 

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
:	ђ`:$ 

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
:	ђ`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:

_output_shapes
: 
р
†
while_cond_9442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9442___redundant_placeholder02
.while_while_cond_9442___redundant_placeholder12
.while_while_cond_9442___redundant_placeholder22
.while_while_cond_9442___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
 S
е
=__inference_gru_layer_call_and_return_conditional_losses_9519

inputs
gru_cell_9431:` 
gru_cell_9433:	ђ`
gru_cell_9435: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ gru_cell/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_mask2
strided_slice_2џ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9431gru_cell_9433gru_cell_9435*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_93762"
 gru_cell/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter‘
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9431gru_cell_9433gru_cell_9435*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_9443*
condR
while_cond_9442*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeљ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_9433*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul–
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_9435*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulХ
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
А
і
#__inference_gru_layer_call_fn_12998
inputs_0
unknown:`
	unknown_0:	ђ`
	unknown_1: `
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_91872
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
"
_user_specified_name
inputs/0
О'
µ
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10148

inputs
	gru_10092:`
	gru_10094:	ђ`
	gru_10096: `
output_10130: 
output_10132:
identityИҐgru/StatefulPartitionedCallҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐoutput/StatefulPartitionedCallв
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_masking_layer_call_and_return_conditional_losses_97952
masking/PartitionedCall±
gru/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0	gru_10092	gru_10094	gru_10096*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_100912
gru/StatefulPartitionedCallЈ
output/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0output_10130output_10132*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_101292 
output/StatefulPartitionedCallє
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10094*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulћ
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp	gru_10096*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЅ
IdentityIdentity'output/StatefulPartitionedCall:output:0^gru/StatefulPartitionedCall6^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
ш!
О
while_body_9111
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0'
while_gru_cell_9133_0:`(
while_gru_cell_9135_0:	ђ`'
while_gru_cell_9137_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor%
while_gru_cell_9133:`&
while_gru_cell_9135:	ђ`%
while_gru_cell_9137: `ИҐ&while/gru_cell/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape‘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЬ
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9133_0while_gru_cell_9135_0while_gru_cell_9137_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_90982(
&while/gru_cell/StatefulPartitionedCallу
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
while/add_1З
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/IdentityЪ
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1Й
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2ґ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Љ
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4",
while_gru_cell_9133while_gru_cell_9133_0",
while_gru_cell_9135while_gru_cell_9135_0",
while_gru_cell_9137while_gru_cell_9137_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
н‘
”
gru_while_body_10904$
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
.gru_while_gru_cell_readvariableop_1_resource_0:	ђ`@
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
,gru_while_gru_cell_readvariableop_1_resource:	ђ`>
,gru_while_gru_cell_readvariableop_4_resource: `ИҐ!gru/while/gru_cell/ReadVariableOpҐ#gru/while/gru_cell/ReadVariableOp_1Ґ#gru/while/gru_cell/ReadVariableOp_2Ґ#gru/while/gru_cell/ReadVariableOp_3Ґ#gru/while/gru_cell/ReadVariableOp_4Ґ#gru/while/gru_cell/ReadVariableOp_5Ґ#gru/while/gru_cell/ReadVariableOp_6Ћ
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeм
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItemѕ
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2?
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeх
/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0gru_while_placeholderFgru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
21
/gru/while/TensorArrayV2Read_1/TensorListGetItemђ
"gru/while/gru_cell/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/ShapeН
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"gru/while/gru_cell/ones_like/Const—
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/ones_likeУ
$gru/while/gru_cell/ones_like_1/ShapeShapegru_while_placeholder_3*
T0*
_output_shapes
:2&
$gru/while/gru_cell/ones_like_1/ShapeС
$gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$gru/while/gru_cell/ones_like_1/ConstЎ
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
gru/while/gru_cell/ones_like_1≥
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
gru/while/gru_cell/unstack«
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/mulЋ
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/mul_1Ћ
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/mul_2Ї
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1°
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack•
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice/stack_1•
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2с
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_sliceє
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMulЇ
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2•
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
*gru/while/gru_cell/strided_slice_1/stack_2ы
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1Ѕ
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_1Ї
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3•
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
*gru/while/gru_cell/strided_slice_2/stack_2ы
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2Ѕ
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_2Ю
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stackҐ
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_3/stack_1Ґ
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2ё
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_3«
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAddЮ
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_4/stackҐ
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_4/stack_1Ґ
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2ћ
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_4Ќ
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_1Ю
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2*
(gru/while/gru_cell/strided_slice_5/stackҐ
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1Ґ
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2№
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2$
"gru/while/gru_cell/strided_slice_5Ќ
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_2ѓ
gru/while/gru_cell/mul_3Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_3ѓ
gru/while/gru_cell/mul_4Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_4ѓ
gru/while/gru_cell/mul_5Mulgru_while_placeholder_3'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_5є
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4•
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
*gru/while/gru_cell/strided_slice_6/stack_2ъ
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6Ѕ
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_3є
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5•
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
*gru/while/gru_cell/strided_slice_7/stack_2ъ
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7Ѕ
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_4Ю
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stackҐ
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_8/stack_1Ґ
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2ё
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_8Ќ
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_3Ю
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_9/stackҐ
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_9/stack_1Ґ
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2ћ
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_9Ќ
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_4Ј
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/addС
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/Sigmoidљ
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/add_1Ч
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/Sigmoid_1є
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6І
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)gru/while/gru_cell/strided_slice_10/stackЂ
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1Ђ
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2€
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10¬
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_5†
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2+
)gru/while/gru_cell/strided_slice_11/stack§
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1§
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2б
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2%
#gru/while/gru_cell/strided_slice_11ќ
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_5ґ
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_6і
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/add_2К
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/Tanh¶
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru/while/gru_cell/sub/xђ
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/sub¶
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_8Ђ
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/add_3Е
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
:€€€€€€€€€2
gru/while/Tileґ
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/SelectV2Й
gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile_1/multiplesї
gru/while/Tile_1Tile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0#gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
gru/while/Tile_1Љ
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/SelectV2_1п
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
gru/while/add_1/yК
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1т
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/IdentityЙ
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1ф
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2°
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3П
gru/while/Identity_4Identitygru/while/SelectV2:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/Identity_4С
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
gru_while_identity_5gru/while/Identity_5:output:0"ј
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0"Є
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : 2F
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Шќ
¬
>__inference_gru_layer_call_and_return_conditional_losses_10091

inputs2
 gru_cell_readvariableop_resource:`5
"gru_cell_readvariableop_1_resource:	ђ`4
"gru_cell_readvariableop_4_resource: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐgru_cell/ReadVariableOpҐgru_cell/ReadVariableOp_1Ґgru_cell/ReadVariableOp_2Ґgru_cell/ReadVariableOp_3Ґgru_cell/ReadVariableOp_4Ґgru_cell/ReadVariableOp_5Ґgru_cell/ReadVariableOp_6ҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
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
 *  А?2
gru_cell/ones_like/Const©
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
gru_cell/ones_like_1/Const∞
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/ones_like_1У
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOpЕ
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstackН
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mulС
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_1С
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru_cell/mul_2Ъ
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_1Н
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stackС
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice/stack_1С
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
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_sliceС
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMulЪ
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_2С
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_1/stackХ
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_1/stack_1Х
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2њ
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_1Щ
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_1Ъ
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
gru_cell/ReadVariableOp_3С
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2 
gru_cell/strided_slice_2/stackХ
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1Х
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2њ
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
gru_cell/strided_slice_2Щ
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_2К
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stackО
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_3/stack_1О
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2Ґ
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_3Я
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAddК
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_4/stackО
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_4/stack_1О
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2Р
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_4•
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_1К
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2 
gru_cell/strided_slice_5/stackО
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1О
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2†
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_5•
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_2И
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_3И
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_4И
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_5Щ
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_4С
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stackХ
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_6/stack_1Х
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2Њ
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_6Щ
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_3Щ
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_5С
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_7/stackХ
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2"
 gru_cell/strided_slice_7/stack_1Х
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2Њ
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_7Щ
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_4К
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stackО
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_8/stack_1О
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2Ґ
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
gru_cell/strided_slice_8•
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_3К
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_9/stackО
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2"
 gru_cell/strided_slice_9/stack_1О
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2Р
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
gru_cell/strided_slice_9•
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_4П
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/SigmoidХ
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Sigmoid_1Щ
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02
gru_cell/ReadVariableOp_6У
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
gru_cell/strided_slice_10/stackЧ
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1Ч
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2√
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
gru_cell/strided_slice_10Ъ
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/MatMul_5М
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2!
gru_cell/strided_slice_11/stackР
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1Р
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2•
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
gru_cell/strided_slice_11¶
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/BiasAdd_5О
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_6М
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru_cell/sub/xД
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/mul_8Г
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru_cell/add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterС
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_9927*
condR
while_cond_9926*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime“
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mulе
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulі
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
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
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
ы
≤
#__inference_gru_layer_call_fn_13020

inputs
unknown:`
	unknown_0:	ђ`
	unknown_1: `
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_100912
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
 S
е
=__inference_gru_layer_call_and_return_conditional_losses_9187

inputs
gru_cell_9099:` 
gru_cell_9101:	ђ`
gru_cell_9103: `
identityИҐ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpҐ gru_cell/StatefulPartitionedCallҐwhileD
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
strided_slice/stack_2в
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
B :и2
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
zeros/packed/1Г
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
:€€€€€€€€€ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2
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
strided_slice_1/stack_2о
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
TensorArrayV2/element_shape≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2њ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeш
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
strided_slice_2/stack_2э
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
shrink_axis_mask2
strided_slice_2џ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9099gru_cell_9101gru_cell_9103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_90982"
 gru_cell/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2
TensorArrayV2_1/element_shapeЄ
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
€€€€€€€€€2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter‘
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9099gru_cell_9101gru_cell_9103*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_9111*
condR
while_cond_9110*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЃ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeљ
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_9101*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul–
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_9103*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulХ
IdentityIdentitytranspose_1:y:06^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
 
_user_specified_nameinputs
іЕ
А
B__inference_gru_cell_layer_call_and_return_conditional_losses_9098

inputs

states)
readvariableop_resource:`,
readvariableop_1_resource:	ђ`+
readvariableop_4_resource: `
identity

identity_1ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3ҐReadVariableOp_4ҐReadVariableOp_5ҐReadVariableOp_6Ґ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
 *  А?2
ones_like/ConstЕ
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
ones_like_1/ConstМ
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ђ2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
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
strided_slice/stack_2€
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Й
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2Й
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
strided_slice_3/stack_2м
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
:€€€€€€€€€ 2	
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
strided_slice_4/stack_2Џ
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4Б
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_5/stack_2к
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5Б
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2e
mul_3Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_8/stack_2м
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8Б
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_9/stack_2Џ
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9Б
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6Б
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2Н
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
:€€€€€€€€€ 2

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
strided_slice_11/stack_2п
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11В
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_3…
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul№
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЏ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
в
т
.__inference_GRU_classifier_layer_call_fn_10161	
input
unknown:`
	unknown_0:	ђ`
	unknown_1: `
	unknown_2: 
	unknown_3:
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *1J 8В *R
fMRK
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_101482
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€€€€€€€€€€ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_nameinput
≥ї
Г
C__inference_gru_cell_layer_call_and_return_conditional_losses_13358

inputs
states_0)
readvariableop_resource:`,
readvariableop_1_resource:	ђ`+
readvariableop_4_resource: `
identity

identity_1ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3ҐReadVariableOp_4ҐReadVariableOp_5ҐReadVariableOp_6Ґ5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpҐ?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpX
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
 *  А?2
ones_like/ConstЕ
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout/ConstА
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape—
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2—ІЉ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_1/ConstЖ
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape„
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ї№ґ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_1/GreaterEqual/y«
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/GreaterEqualЖ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/CastГ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_2/ConstЖ
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape„
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2Йяо2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_2/GreaterEqual/y«
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
dropout_2/GreaterEqualЖ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2
dropout_2/CastГ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
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
 *  А?2
ones_like_1/ConstМ
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ones_like_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_3/ConstЗ
dropout_3/MulMulones_like_1:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/Shape÷
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ЧЊМ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_3/GreaterEqual/y∆
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_3/Mul_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_4/ConstЗ
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape÷
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2Й∆Ы2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_4/GreaterEqual/y∆
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/GreaterEqualЕ
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/CastВ
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2
dropout_5/ConstЗ
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape÷
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2Пд®2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2
dropout_5/GreaterEqual/y∆
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/GreaterEqualЕ
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/CastВ
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
:€€€€€€€€€ђ2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
mul_2
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
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
strided_slice/stack_2€
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Й
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2Й
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
strided_slice_3/stack_2м
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
:€€€€€€€€€ 2	
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
strided_slice_4/stack_2Џ
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_4Б
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_5/stack_2к
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_5Б
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2И
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
:€€€€€€€€€ 2

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
strided_slice_8/stack_2м
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
strided_slice_8Б
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
strided_slice_9/stack_2Џ
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2
strided_slice_9Б
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1~
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02
ReadVariableOp_6Б
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2Н
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
:€€€€€€€€€ 2

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
strided_slice_11/stack_2п
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
strided_slice_11В
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_3…
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:	ђ`*
dtype027
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp√
&gru/gru_cell/kernel/Regularizer/SquareSquare=gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ`2(
&gru/gru_cell/kernel/Regularizer/SquareЯ
%gru/gru_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%gru/gru_cell/kernel/Regularizer/Constќ
#gru/gru_cell/kernel/Regularizer/SumSum*gru/gru_cell/kernel/Regularizer/Square:y:0.gru/gru_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/SumУ
%gru/gru_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2'
%gru/gru_cell/kernel/Regularizer/mul/x–
#gru/gru_cell/kernel/Regularizer/mulMul.gru/gru_cell/kernel/Regularizer/mul/x:output:0,gru/gru_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#gru/gru_cell/kernel/Regularizer/mul№
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_4_resource*
_output_shapes

: `*
dtype02A
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpа
0gru/gru_cell/recurrent_kernel/Regularizer/SquareSquareGgru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: `22
0gru/gru_cell/recurrent_kernel/Regularizer/Square≥
/gru/gru_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       21
/gru/gru_cell/recurrent_kernel/Regularizer/Constц
-gru/gru_cell/recurrent_kernel/Regularizer/SumSum4gru/gru_cell/recurrent_kernel/Regularizer/Square:y:08gru/gru_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/SumІ
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<21
/gru/gru_cell/recurrent_kernel/Regularizer/mul/xш
-gru/gru_cell/recurrent_kernel/Regularizer/mulMul8gru/gru_cell/recurrent_kernel/Regularizer/mul/x:output:06gru/gru_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2/
-gru/gru_cell/recurrent_kernel/Regularizer/mulЏ
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityё

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_66^gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp@^gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€ђ:€€€€€€€€€ : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62n
5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp5gru/gru_cell/kernel/Regularizer/Square/ReadVariableOp2В
?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp?gru/gru_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
Р°
”
gru_while_body_11304$
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
.gru_while_gru_cell_readvariableop_1_resource_0:	ђ`@
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
,gru_while_gru_cell_readvariableop_1_resource:	ђ`>
,gru_while_gru_cell_readvariableop_4_resource: `ИҐ!gru/while/gru_cell/ReadVariableOpҐ#gru/while/gru_cell/ReadVariableOp_1Ґ#gru/while/gru_cell/ReadVariableOp_2Ґ#gru/while/gru_cell/ReadVariableOp_3Ґ#gru/while/gru_cell/ReadVariableOp_4Ґ#gru/while/gru_cell/ReadVariableOp_5Ґ#gru/while/gru_cell/ReadVariableOp_6Ћ
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€,  2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeм
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€ђ*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItemѕ
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2?
=gru/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeх
/gru/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0gru_while_placeholderFgru/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
21
/gru/while/TensorArrayV2Read_1/TensorListGetItemђ
"gru/while/gru_cell/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/ShapeН
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2$
"gru/while/gru_cell/ones_like/Const—
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/ones_likeЙ
 gru/while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2"
 gru/while/gru_cell/dropout/Constћ
gru/while/gru_cell/dropout/MulMul%gru/while/gru_cell/ones_like:output:0)gru/while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2 
gru/while/gru_cell/dropout/MulЩ
 gru/while/gru_cell/dropout/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 gru/while/gru_cell/dropout/ShapeК
7gru/while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform)gru/while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2‘Р 29
7gru/while/gru_cell/dropout/random_uniform/RandomUniformЫ
)gru/while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2+
)gru/while/gru_cell/dropout/GreaterEqual/yЛ
'gru/while/gru_cell/dropout/GreaterEqualGreaterEqual@gru/while/gru_cell/dropout/random_uniform/RandomUniform:output:02gru/while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2)
'gru/while/gru_cell/dropout/GreaterEqualє
gru/while/gru_cell/dropout/CastCast+gru/while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2!
gru/while/gru_cell/dropout/Cast«
 gru/while/gru_cell/dropout/Mul_1Mul"gru/while/gru_cell/dropout/Mul:z:0#gru/while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2"
 gru/while/gru_cell/dropout/Mul_1Н
"gru/while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2$
"gru/while/gru_cell/dropout_1/Const“
 gru/while/gru_cell/dropout_1/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2"
 gru/while/gru_cell/dropout_1/MulЭ
"gru/while/gru_cell/dropout_1/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_1/ShapeП
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2ЇФQ2;
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniformЯ
+gru/while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2-
+gru/while/gru_cell/dropout_1/GreaterEqual/yУ
)gru/while/gru_cell/dropout_1/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_1/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2+
)gru/while/gru_cell/dropout_1/GreaterEqualњ
!gru/while/gru_cell/dropout_1/CastCast-gru/while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2#
!gru/while/gru_cell/dropout_1/Castѕ
"gru/while/gru_cell/dropout_1/Mul_1Mul$gru/while/gru_cell/dropout_1/Mul:z:0%gru/while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2$
"gru/while/gru_cell/dropout_1/Mul_1Н
"gru/while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2$
"gru/while/gru_cell/dropout_2/Const“
 gru/while/gru_cell/dropout_2/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2"
 gru/while/gru_cell/dropout_2/MulЭ
"gru/while/gru_cell/dropout_2/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_2/ShapeП
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ*
dtype0*

seedJ*
seed2Џ¬t2;
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniformЯ
+gru/while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2-
+gru/while/gru_cell/dropout_2/GreaterEqual/yУ
)gru/while/gru_cell/dropout_2/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_2/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2+
)gru/while/gru_cell/dropout_2/GreaterEqualњ
!gru/while/gru_cell/dropout_2/CastCast-gru/while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ђ2#
!gru/while/gru_cell/dropout_2/Castѕ
"gru/while/gru_cell/dropout_2/Mul_1Mul$gru/while/gru_cell/dropout_2/Mul:z:0%gru/while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2$
"gru/while/gru_cell/dropout_2/Mul_1У
$gru/while/gru_cell/ones_like_1/ShapeShapegru_while_placeholder_3*
T0*
_output_shapes
:2&
$gru/while/gru_cell/ones_like_1/ShapeС
$gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$gru/while/gru_cell/ones_like_1/ConstЎ
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
gru/while/gru_cell/ones_like_1Н
"gru/while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2$
"gru/while/gru_cell/dropout_3/Const”
 gru/while/gru_cell/dropout_3/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 gru/while/gru_cell/dropout_3/MulЯ
"gru/while/gru_cell/dropout_3/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_3/ShapeП
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2Й№Ш2;
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniformЯ
+gru/while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2-
+gru/while/gru_cell/dropout_3/GreaterEqual/yТ
)gru/while/gru_cell/dropout_3/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_3/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)gru/while/gru_cell/dropout_3/GreaterEqualЊ
!gru/while/gru_cell/dropout_3/CastCast-gru/while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!gru/while/gru_cell/dropout_3/Castќ
"gru/while/gru_cell/dropout_3/Mul_1Mul$gru/while/gru_cell/dropout_3/Mul:z:0%gru/while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"gru/while/gru_cell/dropout_3/Mul_1Н
"gru/while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2$
"gru/while/gru_cell/dropout_4/Const”
 gru/while/gru_cell/dropout_4/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 gru/while/gru_cell/dropout_4/MulЯ
"gru/while/gru_cell/dropout_4/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_4/ShapeП
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2§÷Б2;
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniformЯ
+gru/while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2-
+gru/while/gru_cell/dropout_4/GreaterEqual/yТ
)gru/while/gru_cell/dropout_4/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_4/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)gru/while/gru_cell/dropout_4/GreaterEqualЊ
!gru/while/gru_cell/dropout_4/CastCast-gru/while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!gru/while/gru_cell/dropout_4/Castќ
"gru/while/gru_cell/dropout_4/Mul_1Mul$gru/while/gru_cell/dropout_4/Mul:z:0%gru/while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"gru/while/gru_cell/dropout_4/Mul_1Н
"gru/while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †@2$
"gru/while/gru_cell/dropout_5/Const”
 gru/while/gru_cell/dropout_5/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 gru/while/gru_cell/dropout_5/MulЯ
"gru/while/gru_cell/dropout_5/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_5/ShapeП
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seedJ*
seed2ЪО—2;
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniformЯ
+gru/while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL?2-
+gru/while/gru_cell/dropout_5/GreaterEqual/yТ
)gru/while/gru_cell/dropout_5/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_5/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)gru/while/gru_cell/dropout_5/GreaterEqualЊ
!gru/while/gru_cell/dropout_5/CastCast-gru/while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2#
!gru/while/gru_cell/dropout_5/Castќ
"gru/while/gru_cell/dropout_5/Mul_1Mul$gru/while/gru_cell/dropout_5/Mul:z:0%gru/while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"gru/while/gru_cell/dropout_5/Mul_1≥
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
gru/while/gru_cell/unstack∆
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0$gru/while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/mulћ
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/mul_1ћ
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ђ2
gru/while/gru_cell/mul_2Ї
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1°
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack•
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice/stack_1•
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2с
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_sliceє
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMulЇ
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2•
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
*gru/while/gru_cell/strided_slice_1/stack_2ы
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1Ѕ
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_1Ї
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:	ђ`*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3•
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
*gru/while/gru_cell/strided_slice_2/stack_2ы
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ђ *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2Ѕ
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_2Ю
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stackҐ
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_3/stack_1Ґ
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2ё
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_3«
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAddЮ
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_4/stackҐ
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_4/stack_1Ґ
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2ћ
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_4Ќ
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_1Ю
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:@2*
(gru/while/gru_cell/strided_slice_5/stackҐ
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1Ґ
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2№
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2$
"gru/while/gru_cell/strided_slice_5Ќ
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_2Ѓ
gru/while/gru_cell/mul_3Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_3Ѓ
gru/while/gru_cell/mul_4Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_4Ѓ
gru/while/gru_cell/mul_5Mulgru_while_placeholder_3&gru/while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_5є
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4•
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
*gru/while/gru_cell/strided_slice_6/stack_2ъ
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6Ѕ
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_3є
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5•
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
*gru/while/gru_cell/strided_slice_7/stack_2ъ
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7Ѕ
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_4Ю
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stackҐ
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_8/stack_1Ґ
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2ё
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2$
"gru/while/gru_cell/strided_slice_8Ќ
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_3Ю
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_9/stackҐ
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:@2,
*gru/while/gru_cell/strided_slice_9/stack_1Ґ
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2ћ
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: 2$
"gru/while/gru_cell/strided_slice_9Ќ
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_4Ј
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/addС
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/Sigmoidљ
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/add_1Ч
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/Sigmoid_1є
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes

: `*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6І
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)gru/while/gru_cell/strided_slice_10/stackЂ
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1Ђ
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2€
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10¬
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/MatMul_5†
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:@2+
)gru/while/gru_cell/strided_slice_11/stack§
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1§
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2б
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2%
#gru/while/gru_cell/strided_slice_11ќ
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/BiasAdd_5ґ
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_6і
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/add_2К
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/Tanh¶
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
gru/while/gru_cell/sub/xђ
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/sub¶
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/mul_8Ђ
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/gru_cell/add_3Е
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
:€€€€€€€€€2
gru/while/Tileґ
gru/while/SelectV2SelectV2gru/while/Tile:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/SelectV2Й
gru/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
gru/while/Tile_1/multiplesї
gru/while/Tile_1Tile6gru/while/TensorArrayV2Read_1/TensorListGetItem:item:0#gru/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:€€€€€€€€€2
gru/while/Tile_1Љ
gru/while/SelectV2_1SelectV2gru/while/Tile_1:output:0gru/while/gru_cell/add_3:z:0gru_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/SelectV2_1п
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
gru/while/add_1/yК
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1т
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/IdentityЙ
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1ф
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2°
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3П
gru/while/Identity_4Identitygru/while/SelectV2:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
gru/while/Identity_4С
gru/while/Identity_5Identitygru/while/SelectV2_1:output:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
gru_while_identity_5gru/while/Identity_5:output:0"ј
]gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_gru_while_tensorarrayv2read_1_tensorlistgetitem_gru_tensorarrayunstack_1_tensorlistfromtensor_0"Є
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : : 2F
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
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
х
•
while_cond_12431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_12431___redundant_placeholder03
/while_while_cond_12431___redundant_placeholder13
/while_while_cond_12431___redundant_placeholder23
/while_while_cond_12431___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ј
serving_defaultђ
E
input<
serving_default_input:0€€€€€€€€€€€€€€€€€€ђG
output=
StatefulPartitionedCall:0€€€€€€€€€€€€€€€€€€tensorflow/serving/predict:щ»
ш0
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
*V&call_and_return_all_conditional_losses
W_default_save_signature
X__call__"≈.
_tf_keras_network©.{"name": "GRU_classifier", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "GRU_classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "name": "masking", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["masking", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["gru", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 300]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "GRU_classifier", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "name": "masking", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["masking", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["gru", 0, 0, {}]]], "shared_object_id": 10}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 13}, "metrics": [[{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 14}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00039999998989515007, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
х"т
_tf_keras_input_layer“{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
З
	variables
trainable_variables
regularization_losses
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"ш
_tf_keras_layerё{"name": "masking", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "stateful": false, "must_restore_from_config": false, "class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "dtype": "float32", "mask_value": 0.0}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}
т
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"…
_tf_keras_rnn_layerЂ{"name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "inbound_nodes": [[["masking", 0, 0, {}]]], "shared_object_id": 7, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}}
ы

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"÷
_tf_keras_layerЉ{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["gru", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 32]}}
≠
iter

beta_1

beta_2
	decay
learning_ratemLmM mN!mO"mPvQvR vS!vT"vU"
	optimizer
C
 0
!1
"2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
 0
!1
"2
3
4"
trackable_list_wrapper
 
#non_trainable_variables

$layers
%layer_regularization_losses
	variables
&layer_metrics
'metrics
regularization_losses
trainable_variables
X__call__
W_default_save_signature
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
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
≠
(non_trainable_variables

)layers
*layer_regularization_losses
	variables
+layer_metrics
trainable_variables
,metrics
regularization_losses
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Ц


 kernel
!recurrent_kernel
"bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
*`&call_and_return_all_conditional_losses
a__call__"џ
_tf_keras_layerЅ{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 6}
 "
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
.
b0
c1"
trackable_list_wrapper
є

1layers
2non_trainable_variables
3layer_regularization_losses
	variables
4layer_metrics
trainable_variables
5metrics
regularization_losses

6states
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
: 2output/kernel
:2output/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
7non_trainable_variables

8layers
9layer_regularization_losses
	variables
:layer_metrics
trainable_variables
;metrics
regularization_losses
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
&:$	ђ`2gru/gru_cell/kernel
/:- `2gru/gru_cell/recurrent_kernel
#:!`2gru/gru_cell/bias
 "
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
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
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
.
b0
c1"
trackable_list_wrapper
≠
>non_trainable_variables

?layers
@layer_regularization_losses
-	variables
Alayer_metrics
.trainable_variables
Bmetrics
/regularization_losses
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
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
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
‘
	Ctotal
	Dcount
E	variables
F	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 17}
І
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"а
_tf_keras_metric≈{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "shared_object_id": 14}
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
 "
trackable_list_wrapper
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
+:)	ђ`2Adam/gru/gru_cell/kernel/m
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/m
(:&`2Adam/gru/gru_cell/bias/m
$:" 2Adam/output/kernel/v
:2Adam/output/bias/v
+:)	ђ`2Adam/gru/gru_cell/kernel/v
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/v
(:&`2Adam/gru/gru_cell/bias/v
т2п
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_11109
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_11557
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10693
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10722ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
й2ж
__inference__wrapped_model_8949¬
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *2Ґ/
-К*
input€€€€€€€€€€€€€€€€€€ђ
Ж2Г
.__inference_GRU_classifier_layer_call_fn_10161
.__inference_GRU_classifier_layer_call_fn_11572
.__inference_GRU_classifier_layer_call_fn_11587
.__inference_GRU_classifier_layer_call_fn_10664ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_masking_layer_call_and_return_conditional_losses_11598Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_masking_layer_call_fn_11603Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
џ2Ў
>__inference_gru_layer_call_and_return_conditional_losses_11910
>__inference_gru_layer_call_and_return_conditional_losses_12301
>__inference_gru_layer_call_and_return_conditional_losses_12596
>__inference_gru_layer_call_and_return_conditional_losses_12987’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
п2м
#__inference_gru_layer_call_fn_12998
#__inference_gru_layer_call_fn_13009
#__inference_gru_layer_call_fn_13020
#__inference_gru_layer_call_fn_13031’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
л2и
A__inference_output_layer_call_and_return_conditional_losses_13061Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_output_layer_call_fn_13070Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»B≈
#__inference_signature_wrapper_10757input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
C__inference_gru_cell_layer_call_and_return_conditional_losses_13196
C__inference_gru_cell_layer_call_and_return_conditional_losses_13358Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ш2Х
(__inference_gru_cell_layer_call_fn_13372
(__inference_gru_cell_layer_call_fn_13386Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≤2ѓ
__inference_loss_fn_0_13397П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤2ѓ
__inference_loss_fn_1_13408П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ ѕ
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10693Б" !DҐA
:Ґ7
-К*
input€€€€€€€€€€€€€€€€€€ђ
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѕ
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_10722Б" !DҐA
:Ґ7
-К*
input€€€€€€€€€€€€€€€€€€ђ
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ –
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_11109В" !EҐB
;Ґ8
.К+
inputs€€€€€€€€€€€€€€€€€€ђ
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ –
I__inference_GRU_classifier_layer_call_and_return_conditional_losses_11557В" !EҐB
;Ґ8
.К+
inputs€€€€€€€€€€€€€€€€€€ђ
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ¶
.__inference_GRU_classifier_layer_call_fn_10161t" !DҐA
:Ґ7
-К*
input€€€€€€€€€€€€€€€€€€ђ
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€¶
.__inference_GRU_classifier_layer_call_fn_10664t" !DҐA
:Ґ7
-К*
input€€€€€€€€€€€€€€€€€€ђ
p

 
™ "%К"€€€€€€€€€€€€€€€€€€І
.__inference_GRU_classifier_layer_call_fn_11572u" !EҐB
;Ґ8
.К+
inputs€€€€€€€€€€€€€€€€€€ђ
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€І
.__inference_GRU_classifier_layer_call_fn_11587u" !EҐB
;Ґ8
.К+
inputs€€€€€€€€€€€€€€€€€€ђ
p

 
™ "%К"€€€€€€€€€€€€€€€€€€І
__inference__wrapped_model_8949Г" !<Ґ9
2Ґ/
-К*
input€€€€€€€€€€€€€€€€€€ђ
™ "<™9
7
output-К*
output€€€€€€€€€€€€€€€€€€А
C__inference_gru_cell_layer_call_and_return_conditional_losses_13196Є" !]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€ 
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€ 
$Ъ!
К
0/1/0€€€€€€€€€ 
Ъ А
C__inference_gru_cell_layer_call_and_return_conditional_losses_13358Є" !]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€ 
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€ 
$Ъ!
К
0/1/0€€€€€€€€€ 
Ъ „
(__inference_gru_cell_layer_call_fn_13372™" !]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€ 
p 
™ "DҐA
К
0€€€€€€€€€ 
"Ъ
К
1/0€€€€€€€€€ „
(__inference_gru_cell_layer_call_fn_13386™" !]ҐZ
SҐP
!К
inputs€€€€€€€€€ђ
'Ґ$
"К
states/0€€€€€€€€€ 
p
™ "DҐA
К
0€€€€€€€€€ 
"Ъ
К
1/0€€€€€€€€€ ќ
>__inference_gru_layer_call_and_return_conditional_losses_11910Л" !PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ ќ
>__inference_gru_layer_call_and_return_conditional_losses_12301Л" !PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ «
>__inference_gru_layer_call_and_return_conditional_losses_12596Д" !IҐF
?Ґ<
.К+
inputs€€€€€€€€€€€€€€€€€€ђ

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ «
>__inference_gru_layer_call_and_return_conditional_losses_12987Д" !IҐF
?Ґ<
.К+
inputs€€€€€€€€€€€€€€€€€€ђ

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ •
#__inference_gru_layer_call_fn_12998~" !PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€ •
#__inference_gru_layer_call_fn_13009~" !PҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€ђ

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€ Ю
#__inference_gru_layer_call_fn_13020w" !IҐF
?Ґ<
.К+
inputs€€€€€€€€€€€€€€€€€€ђ

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€ Ю
#__inference_gru_layer_call_fn_13031w" !IҐF
?Ґ<
.К+
inputs€€€€€€€€€€€€€€€€€€ђ

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€ :
__inference_loss_fn_0_13397 Ґ

Ґ 
™ "К :
__inference_loss_fn_1_13408!Ґ

Ґ 
™ "К Ї
B__inference_masking_layer_call_and_return_conditional_losses_11598t=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€€€€€€€€€€ђ
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€ђ
Ъ Т
'__inference_masking_layer_call_fn_11603g=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€€€€€€€€€€ђ
™ "&К#€€€€€€€€€€€€€€€€€€ђї
A__inference_output_layer_call_and_return_conditional_losses_13061v<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€ 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ У
&__inference_output_layer_call_fn_13070i<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€ 
™ "%К"€€€€€€€€€€€€€€€€€€і
#__inference_signature_wrapper_10757М" !EҐB
Ґ 
;™8
6
input-К*
input€€€€€€€€€€€€€€€€€€ђ"<™9
7
output-К*
output€€€€€€€€€€€€€€€€€€
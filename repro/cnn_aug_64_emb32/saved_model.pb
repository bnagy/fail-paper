??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
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
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
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
delete_old_dirsbool(?

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
?
Adam/classify/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/classify/bias/v
y
(Adam/classify/bias/v/Read/ReadVariableOpReadVariableOpAdam/classify/bias/v*
_output_shapes
:	*
dtype0
?
Adam/classify/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*'
shared_nameAdam/classify/kernel/v
?
*Adam/classify/kernel/v/Read/ReadVariableOpReadVariableOpAdam/classify/kernel/v*
_output_shapes

:@	*
dtype0
|
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense2/bias/v
u
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*%
shared_nameAdam/dense2/kernel/v
}
(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v*
_output_shapes

:@@*
dtype0
|
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/v
u
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*%
shared_nameAdam/dense1/kernel/v
~
(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v*
_output_shapes
:	?@*
dtype0
z
Adam/norm2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/norm2/beta/v
s
%Adam/norm2/beta/v/Read/ReadVariableOpReadVariableOpAdam/norm2/beta/v*
_output_shapes
:0*
dtype0
|
Adam/norm2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameAdam/norm2/gamma/v
u
&Adam/norm2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/norm2/gamma/v*
_output_shapes
:0*
dtype0
z
Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/conv2/bias/v
s
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/conv2/kernel/v
?
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*&
_output_shapes
:0*
dtype0
z
Adam/norm1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/norm1/beta/v
s
%Adam/norm1/beta/v/Read/ReadVariableOpReadVariableOpAdam/norm1/beta/v*
_output_shapes
:*
dtype0
|
Adam/norm1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/norm1/gamma/v
u
&Adam/norm1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/norm1/gamma/v*
_output_shapes
:*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/conv1/kernel/v
?
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/embed/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?J *(
shared_nameAdam/embed/embeddings/v
?
+Adam/embed/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embed/embeddings/v*
_output_shapes
:	?J *
dtype0
?
Adam/classify/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/classify/bias/m
y
(Adam/classify/bias/m/Read/ReadVariableOpReadVariableOpAdam/classify/bias/m*
_output_shapes
:	*
dtype0
?
Adam/classify/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*'
shared_nameAdam/classify/kernel/m
?
*Adam/classify/kernel/m/Read/ReadVariableOpReadVariableOpAdam/classify/kernel/m*
_output_shapes

:@	*
dtype0
|
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense2/bias/m
u
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*%
shared_nameAdam/dense2/kernel/m
}
(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m*
_output_shapes

:@@*
dtype0
|
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/m
u
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*%
shared_nameAdam/dense1/kernel/m
~
(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m*
_output_shapes
:	?@*
dtype0
z
Adam/norm2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/norm2/beta/m
s
%Adam/norm2/beta/m/Read/ReadVariableOpReadVariableOpAdam/norm2/beta/m*
_output_shapes
:0*
dtype0
|
Adam/norm2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameAdam/norm2/gamma/m
u
&Adam/norm2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/norm2/gamma/m*
_output_shapes
:0*
dtype0
z
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/conv2/bias/m
s
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/conv2/kernel/m
?
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*&
_output_shapes
:0*
dtype0
z
Adam/norm1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/norm1/beta/m
s
%Adam/norm1/beta/m/Read/ReadVariableOpReadVariableOpAdam/norm1/beta/m*
_output_shapes
:*
dtype0
|
Adam/norm1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/norm1/gamma/m
u
&Adam/norm1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/norm1/gamma/m*
_output_shapes
:*
dtype0
z
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/m
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/conv1/kernel/m
?
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/embed/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?J *(
shared_nameAdam/embed/embeddings/m
?
+Adam/embed/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embed/embeddings/m*
_output_shapes
:	?J *
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
r
classify/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameclassify/bias
k
!classify/bias/Read/ReadVariableOpReadVariableOpclassify/bias*
_output_shapes
:	*
dtype0
z
classify/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	* 
shared_nameclassify/kernel
s
#classify/kernel/Read/ReadVariableOpReadVariableOpclassify/kernel*
_output_shapes

:@	*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:@*
dtype0
v
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense2/kernel
o
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes

:@@*
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:@*
dtype0
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	?@*
dtype0
?
norm2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_namenorm2/moving_variance
{
)norm2/moving_variance/Read/ReadVariableOpReadVariableOpnorm2/moving_variance*
_output_shapes
:0*
dtype0
z
norm2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_namenorm2/moving_mean
s
%norm2/moving_mean/Read/ReadVariableOpReadVariableOpnorm2/moving_mean*
_output_shapes
:0*
dtype0
l

norm2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
norm2/beta
e
norm2/beta/Read/ReadVariableOpReadVariableOp
norm2/beta*
_output_shapes
:0*
dtype0
n
norm2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namenorm2/gamma
g
norm2/gamma/Read/ReadVariableOpReadVariableOpnorm2/gamma*
_output_shapes
:0*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:0*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:0*
dtype0
?
norm1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namenorm1/moving_variance
{
)norm1/moving_variance/Read/ReadVariableOpReadVariableOpnorm1/moving_variance*
_output_shapes
:*
dtype0
z
norm1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namenorm1/moving_mean
s
%norm1/moving_mean/Read/ReadVariableOpReadVariableOpnorm1/moving_mean*
_output_shapes
:*
dtype0
l

norm1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
norm1/beta
e
norm1/beta/Read/ReadVariableOpReadVariableOp
norm1/beta*
_output_shapes
:*
dtype0
n
norm1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namenorm1/gamma
g
norm1/gamma/Read/ReadVariableOpReadVariableOpnorm1/gamma*
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
: *
dtype0
}
embed/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?J *!
shared_nameembed/embeddings
v
$embed/embeddings/Read/ReadVariableOpReadVariableOpembed/embeddings*
_output_shapes
:	?J *
dtype0

NoOpNoOp
Ǐ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses* 
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'
embeddings*
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator* 
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance*
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator* 
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op*
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance*
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|_random_generator* 
?
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
'0
;1
<2
K3
L4
M5
N6
\7
]8
l9
m10
n11
o12
?13
?14
?15
?16
?17
?18*
p
;0
<1
K2
L3
\4
]5
l6
m7
?8
?9
?10
?11
?12
?13*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate'm?;m?<m?Km?Lm?\m?]m?lm?mm?	?m?	?m?	?m?	?m?	?m?	?m?'v?;v?<v?Kv?Lv?\v?]v?lv?mv?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

'0*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
d^
VARIABLE_VALUEembed/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

;0
<1*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
K0
L1
M2
N3*

K0
L1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ZT
VARIABLE_VALUEnorm1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
norm1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEnorm1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEnorm1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

\0
]1*

\0
]1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
\V
VARIABLE_VALUEconv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
l0
m1
n2
o3*

l0
m1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ZT
VARIABLE_VALUEnorm2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
norm2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEnorm2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEnorm2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
]W
VARIABLE_VALUEdense2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEclassify/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEclassify/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
'
'0
M1
N2
n3
o4*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

'0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

n0
o1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUEAdam/embed/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/norm1/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/norm1/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/norm2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/norm2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/classify/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/classify/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embed/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/norm1/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/norm1/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/norm2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/norm2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/dense2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/classify/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/classify/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
 serving_default_flatten_40_inputPlaceholder*+
_output_shapes
:?????????@*
dtype0* 
shape:?????????@
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_flatten_40_inputembed/embeddingsconv1/kernel
conv1/biasnorm1/gamma
norm1/betanorm1/moving_meannorm1/moving_varianceconv2/kernel
conv2/biasnorm2/gamma
norm2/betanorm2/moving_meannorm2/moving_variancedense1/kerneldense1/biasdense2/kerneldense2/biasclassify/kernelclassify/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_13240226
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$embed/embeddings/Read/ReadVariableOp conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOpnorm1/gamma/Read/ReadVariableOpnorm1/beta/Read/ReadVariableOp%norm1/moving_mean/Read/ReadVariableOp)norm1/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOpnorm2/gamma/Read/ReadVariableOpnorm2/beta/Read/ReadVariableOp%norm2/moving_mean/Read/ReadVariableOp)norm2/moving_variance/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp#classify/kernel/Read/ReadVariableOp!classify/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/embed/embeddings/m/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp&Adam/norm1/gamma/m/Read/ReadVariableOp%Adam/norm1/beta/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp&Adam/norm2/gamma/m/Read/ReadVariableOp%Adam/norm2/beta/m/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp*Adam/classify/kernel/m/Read/ReadVariableOp(Adam/classify/bias/m/Read/ReadVariableOp+Adam/embed/embeddings/v/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp&Adam/norm1/gamma/v/Read/ReadVariableOp%Adam/norm1/beta/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp&Adam/norm2/gamma/v/Read/ReadVariableOp%Adam/norm2/beta/v/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOp*Adam/classify/kernel/v/Read/ReadVariableOp(Adam/classify/bias/v/Read/ReadVariableOpConst*G
Tin@
>2<	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_13241127
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembed/embeddingsconv1/kernel
conv1/biasnorm1/gamma
norm1/betanorm1/moving_meannorm1/moving_varianceconv2/kernel
conv2/biasnorm2/gamma
norm2/betanorm2/moving_meannorm2/moving_variancedense1/kerneldense1/biasdense2/kerneldense2/biasclassify/kernelclassify/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/embed/embeddings/mAdam/conv1/kernel/mAdam/conv1/bias/mAdam/norm1/gamma/mAdam/norm1/beta/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/norm2/gamma/mAdam/norm2/beta/mAdam/dense1/kernel/mAdam/dense1/bias/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/classify/kernel/mAdam/classify/bias/mAdam/embed/embeddings/vAdam/conv1/kernel/vAdam/conv1/bias/vAdam/norm1/gamma/vAdam/norm1/beta/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/norm2/gamma/vAdam/norm2/beta/vAdam/dense1/kernel/vAdam/dense1/bias/vAdam/dense2/kernel/vAdam/dense2/bias/vAdam/classify/kernel/vAdam/classify/bias/v*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_13241311??
?
?
)__inference_dense1_layer_call_fn_13240854

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_13239613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_norm1_layer_call_and_return_conditional_losses_13240670

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

g
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240715

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? 
*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 
:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?
?
+__inference_classify_layer_call_fn_13240919

inputs
unknown:@	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_classify_layer_call_and_return_conditional_losses_13239653o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_dropout_80_layer_call_fn_13240574

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239525h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@ :W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?

g
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239731

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
n
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13240626

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_cnn_layer_call_fn_13239701
flatten_40_input
unknown:	?J #
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:#
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:	?@

unknown_13:@

unknown_14:@@

unknown_15:@

unknown_16:@	

unknown_17:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_layer_call_and_return_conditional_losses_13239660o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????@
*
_user_specified_nameflatten_40_input
?
?
C__inference_norm2_layer_call_and_return_conditional_losses_13239433

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
C__inference_norm1_layer_call_and_return_conditional_losses_13239388

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_40_layer_call_and_return_conditional_losses_13240533

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????
Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_norm1_layer_call_fn_13240652

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm1_layer_call_and_return_conditional_losses_13239388?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_norm2_layer_call_and_return_conditional_losses_13239464

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
f
-__inference_dropout_82_layer_call_fn_13240828

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239774p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
S
7__inference_average_pooling2d_41_layer_call_fn_13240740

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13239408?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_norm2_layer_call_fn_13240771

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm2_layer_call_and_return_conditional_losses_13239464?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
C__inference_norm1_layer_call_and_return_conditional_losses_13240688

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
n
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13239332

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_41_layer_call_and_return_conditional_losses_13240818

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?_
?
A__inference_cnn_layer_call_and_return_conditional_losses_13240403

inputs2
embed_embedding_lookup_13240318:	?J >
$conv1_conv2d_readvariableop_resource: 3
%conv1_biasadd_readvariableop_resource:+
norm1_readvariableop_resource:-
norm1_readvariableop_1_resource:<
.norm1_fusedbatchnormv3_readvariableop_resource:>
0norm1_fusedbatchnormv3_readvariableop_1_resource:>
$conv2_conv2d_readvariableop_resource:03
%conv2_biasadd_readvariableop_resource:0+
norm2_readvariableop_resource:0-
norm2_readvariableop_1_resource:0<
.norm2_fusedbatchnormv3_readvariableop_resource:0>
0norm2_fusedbatchnormv3_readvariableop_1_resource:08
%dense1_matmul_readvariableop_resource:	?@4
&dense1_biasadd_readvariableop_resource:@7
%dense2_matmul_readvariableop_resource:@@4
&dense2_biasadd_readvariableop_resource:@9
'classify_matmul_readvariableop_resource:@	6
(classify_biasadd_readvariableop_resource:	
identity??classify/BiasAdd/ReadVariableOp?classify/MatMul/ReadVariableOp?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?embed/embedding_lookup?%norm1/FusedBatchNormV3/ReadVariableOp?'norm1/FusedBatchNormV3/ReadVariableOp_1?norm1/ReadVariableOp?norm1/ReadVariableOp_1?%norm2/FusedBatchNormV3/ReadVariableOp?'norm2/FusedBatchNormV3/ReadVariableOp_1?norm2/ReadVariableOp?norm2/ReadVariableOp_1a
flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   s
flatten_40/ReshapeReshapeinputsflatten_40/Const:output:0*
T0*(
_output_shapes
:??????????
q

embed/CastCastflatten_40/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????
?
embed/embedding_lookupResourceGatherembed_embedding_lookup_13240318embed/Cast:y:0*
Tindices0*2
_class(
&$loc:@embed/embedding_lookup/13240318*,
_output_shapes
:??????????
 *
dtype0?
embed/embedding_lookup/IdentityIdentityembed/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embed/embedding_lookup/13240318*,
_output_shapes
:??????????
 ?
!embed/embedding_lookup/Identity_1Identity(embed/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????
 j
reshape_20/ShapeShape*embed/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:h
reshape_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_20/strided_sliceStridedSlicereshape_20/Shape:output:0'reshape_20/strided_slice/stack:output:0)reshape_20/strided_slice/stack_1:output:0)reshape_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@\
reshape_20/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_20/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
reshape_20/Reshape/shapePack!reshape_20/strided_slice:output:0#reshape_20/Reshape/shape/1:output:0#reshape_20/Reshape/shape/2:output:0#reshape_20/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_20/ReshapeReshape*embed/embedding_lookup/Identity_1:output:0!reshape_20/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@ v
dropout_80/IdentityIdentityreshape_20/Reshape:output:0*
T0*/
_output_shapes
:?????????@ ?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv1/Conv2DConv2Ddropout_80/Identity:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
average_pooling2d_40/AvgPoolAvgPoolconv1/Relu:activations:0*
T0*/
_output_shapes
:????????? 
*
ksize
*
paddingVALID*
strides
n
norm1/ReadVariableOpReadVariableOpnorm1_readvariableop_resource*
_output_shapes
:*
dtype0r
norm1/ReadVariableOp_1ReadVariableOpnorm1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp.norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
'norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
norm1/FusedBatchNormV3FusedBatchNormV3%average_pooling2d_40/AvgPool:output:0norm1/ReadVariableOp:value:0norm1/ReadVariableOp_1:value:0-norm1/FusedBatchNormV3/ReadVariableOp:value:0/norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? 
:::::*
epsilon%o?:*
is_training( u
dropout_81/IdentityIdentitynorm1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 
?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2/Conv2DConv2Ddropout_81/Identity:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0*
paddingSAME*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0d

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
0?
average_pooling2d_41/AvgPoolAvgPoolconv2/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
n
norm2/ReadVariableOpReadVariableOpnorm2_readvariableop_resource*
_output_shapes
:0*
dtype0r
norm2/ReadVariableOp_1ReadVariableOpnorm2_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
%norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp.norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
'norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
norm2/FusedBatchNormV3FusedBatchNormV3%average_pooling2d_41/AvgPool:output:0norm2/ReadVariableOp:value:0norm2/ReadVariableOp_1:value:0-norm2/FusedBatchNormV3/ReadVariableOp:value:0/norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
is_training( a
flatten_41/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_41/ReshapeReshapenorm2/FusedBatchNormV3:y:0flatten_41/Const:output:0*
T0*(
_output_shapes
:??????????o
dropout_82/IdentityIdentityflatten_41/Reshape:output:0*
T0*(
_output_shapes
:???????????
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense1/MatMulMatMuldropout_82/Identity:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
dense2/MatMulMatMuldense1/BiasAdd:output:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
dropout_83/IdentityIdentitydense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
classify/MatMul/ReadVariableOpReadVariableOp'classify_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0?
classify/MatMulMatMuldropout_83/Identity:output:0&classify/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
classify/BiasAdd/ReadVariableOpReadVariableOp(classify_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
classify/BiasAddBiasAddclassify/MatMul:product:0'classify/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	h
classify/SoftmaxSoftmaxclassify/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	i
IdentityIdentityclassify/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp ^classify/BiasAdd/ReadVariableOp^classify/MatMul/ReadVariableOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^embed/embedding_lookup&^norm1/FusedBatchNormV3/ReadVariableOp(^norm1/FusedBatchNormV3/ReadVariableOp_1^norm1/ReadVariableOp^norm1/ReadVariableOp_1&^norm2/FusedBatchNormV3/ReadVariableOp(^norm2/FusedBatchNormV3/ReadVariableOp_1^norm2/ReadVariableOp^norm2/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2B
classify/BiasAdd/ReadVariableOpclassify/BiasAdd/ReadVariableOp2@
classify/MatMul/ReadVariableOpclassify/MatMul/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp20
embed/embedding_lookupembed/embedding_lookup2N
%norm1/FusedBatchNormV3/ReadVariableOp%norm1/FusedBatchNormV3/ReadVariableOp2R
'norm1/FusedBatchNormV3/ReadVariableOp_1'norm1/FusedBatchNormV3/ReadVariableOp_12,
norm1/ReadVariableOpnorm1/ReadVariableOp20
norm1/ReadVariableOp_1norm1/ReadVariableOp_12N
%norm2/FusedBatchNormV3/ReadVariableOp%norm2/FusedBatchNormV3/ReadVariableOp2R
'norm2/FusedBatchNormV3/ReadVariableOp_1'norm2/FusedBatchNormV3/ReadVariableOp_12,
norm2/ReadVariableOpnorm2/ReadVariableOp20
norm2/ReadVariableOp_1norm2/ReadVariableOp_1:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

g
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240596

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@ *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@ :W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?
?
C__inference_conv1_layer_call_and_return_conditional_losses_13240616

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?J
?
A__inference_cnn_layer_call_and_return_conditional_losses_13240175
flatten_40_input!
embed_13240120:	?J (
conv1_13240125: 
conv1_13240127:
norm1_13240131:
norm1_13240133:
norm1_13240135:
norm1_13240137:(
conv2_13240141:0
conv2_13240143:0
norm2_13240147:0
norm2_13240149:0
norm2_13240151:0
norm2_13240153:0"
dense1_13240158:	?@
dense1_13240160:@!
dense2_13240163:@@
dense2_13240165:@#
classify_13240169:@	
classify_13240171:	
identity?? classify/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?"dropout_80/StatefulPartitionedCall?"dropout_81/StatefulPartitionedCall?"dropout_82/StatefulPartitionedCall?"dropout_83/StatefulPartitionedCall?embed/StatefulPartitionedCall?norm1/StatefulPartitionedCall?norm2/StatefulPartitionedCall?
flatten_40/PartitionedCallPartitionedCallflatten_40_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_40_layer_call_and_return_conditional_losses_13239488?
embed/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0embed_13240120*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_embed_layer_call_and_return_conditional_losses_13239500?
reshape_20/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_13239518?
"dropout_80/StatefulPartitionedCallStatefulPartitionedCall#reshape_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239846?
conv1/StatefulPartitionedCallStatefulPartitionedCall+dropout_80/StatefulPartitionedCall:output:0conv1_13240125conv1_13240127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_13239538?
$average_pooling2d_40/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13239332?
norm1/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0norm1_13240131norm1_13240133norm1_13240135norm1_13240137*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm1_layer_call_and_return_conditional_losses_13239388?
"dropout_81/StatefulPartitionedCallStatefulPartitionedCall&norm1/StatefulPartitionedCall:output:0#^dropout_80/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239813?
conv2/StatefulPartitionedCallStatefulPartitionedCall+dropout_81/StatefulPartitionedCall:output:0conv2_13240141conv2_13240143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_13239572?
$average_pooling2d_41/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13239408?
norm2/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0norm2_13240147norm2_13240149norm2_13240151norm2_13240153*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm2_layer_call_and_return_conditional_losses_13239464?
flatten_41/PartitionedCallPartitionedCall&norm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_41_layer_call_and_return_conditional_losses_13239594?
"dropout_82/StatefulPartitionedCallStatefulPartitionedCall#flatten_41/PartitionedCall:output:0#^dropout_81/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239774?
dense1/StatefulPartitionedCallStatefulPartitionedCall+dropout_82/StatefulPartitionedCall:output:0dense1_13240158dense1_13240160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_13239613?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_13240163dense2_13240165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_13239629?
"dropout_83/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0#^dropout_82/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239731?
 classify/StatefulPartitionedCallStatefulPartitionedCall+dropout_83/StatefulPartitionedCall:output:0classify_13240169classify_13240171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_classify_layer_call_and_return_conditional_losses_13239653x
IdentityIdentity)classify/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp!^classify/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall#^dropout_80/StatefulPartitionedCall#^dropout_81/StatefulPartitionedCall#^dropout_82/StatefulPartitionedCall#^dropout_83/StatefulPartitionedCall^embed/StatefulPartitionedCall^norm1/StatefulPartitionedCall^norm2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2D
 classify/StatefulPartitionedCall classify/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2H
"dropout_80/StatefulPartitionedCall"dropout_80/StatefulPartitionedCall2H
"dropout_81/StatefulPartitionedCall"dropout_81/StatefulPartitionedCall2H
"dropout_82/StatefulPartitionedCall"dropout_82/StatefulPartitionedCall2H
"dropout_83/StatefulPartitionedCall"dropout_83/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall2>
norm1/StatefulPartitionedCallnorm1/StatefulPartitionedCall2>
norm2/StatefulPartitionedCallnorm2/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????@
*
_user_specified_nameflatten_40_input
?J
?
A__inference_cnn_layer_call_and_return_conditional_losses_13239973

inputs!
embed_13239918:	?J (
conv1_13239923: 
conv1_13239925:
norm1_13239929:
norm1_13239931:
norm1_13239933:
norm1_13239935:(
conv2_13239939:0
conv2_13239941:0
norm2_13239945:0
norm2_13239947:0
norm2_13239949:0
norm2_13239951:0"
dense1_13239956:	?@
dense1_13239958:@!
dense2_13239961:@@
dense2_13239963:@#
classify_13239967:@	
classify_13239969:	
identity?? classify/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?"dropout_80/StatefulPartitionedCall?"dropout_81/StatefulPartitionedCall?"dropout_82/StatefulPartitionedCall?"dropout_83/StatefulPartitionedCall?embed/StatefulPartitionedCall?norm1/StatefulPartitionedCall?norm2/StatefulPartitionedCall?
flatten_40/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_40_layer_call_and_return_conditional_losses_13239488?
embed/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0embed_13239918*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_embed_layer_call_and_return_conditional_losses_13239500?
reshape_20/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_13239518?
"dropout_80/StatefulPartitionedCallStatefulPartitionedCall#reshape_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239846?
conv1/StatefulPartitionedCallStatefulPartitionedCall+dropout_80/StatefulPartitionedCall:output:0conv1_13239923conv1_13239925*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_13239538?
$average_pooling2d_40/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13239332?
norm1/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0norm1_13239929norm1_13239931norm1_13239933norm1_13239935*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm1_layer_call_and_return_conditional_losses_13239388?
"dropout_81/StatefulPartitionedCallStatefulPartitionedCall&norm1/StatefulPartitionedCall:output:0#^dropout_80/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239813?
conv2/StatefulPartitionedCallStatefulPartitionedCall+dropout_81/StatefulPartitionedCall:output:0conv2_13239939conv2_13239941*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_13239572?
$average_pooling2d_41/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13239408?
norm2/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0norm2_13239945norm2_13239947norm2_13239949norm2_13239951*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm2_layer_call_and_return_conditional_losses_13239464?
flatten_41/PartitionedCallPartitionedCall&norm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_41_layer_call_and_return_conditional_losses_13239594?
"dropout_82/StatefulPartitionedCallStatefulPartitionedCall#flatten_41/PartitionedCall:output:0#^dropout_81/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239774?
dense1/StatefulPartitionedCallStatefulPartitionedCall+dropout_82/StatefulPartitionedCall:output:0dense1_13239956dense1_13239958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_13239613?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_13239961dense2_13239963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_13239629?
"dropout_83/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0#^dropout_82/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239731?
 classify/StatefulPartitionedCallStatefulPartitionedCall+dropout_83/StatefulPartitionedCall:output:0classify_13239967classify_13239969*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_classify_layer_call_and_return_conditional_losses_13239653x
IdentityIdentity)classify/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp!^classify/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall#^dropout_80/StatefulPartitionedCall#^dropout_81/StatefulPartitionedCall#^dropout_82/StatefulPartitionedCall#^dropout_83/StatefulPartitionedCall^embed/StatefulPartitionedCall^norm1/StatefulPartitionedCall^norm2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2D
 classify/StatefulPartitionedCall classify/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2H
"dropout_80/StatefulPartitionedCall"dropout_80/StatefulPartitionedCall2H
"dropout_81/StatefulPartitionedCall"dropout_81/StatefulPartitionedCall2H
"dropout_82/StatefulPartitionedCall"dropout_82/StatefulPartitionedCall2H
"dropout_83/StatefulPartitionedCall"dropout_83/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall2>
norm1/StatefulPartitionedCallnorm1/StatefulPartitionedCall2>
norm2/StatefulPartitionedCallnorm2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_flatten_41_layer_call_fn_13240812

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_41_layer_call_and_return_conditional_losses_13239594a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?D
?
A__inference_cnn_layer_call_and_return_conditional_losses_13239660

inputs!
embed_13239501:	?J (
conv1_13239539: 
conv1_13239541:
norm1_13239545:
norm1_13239547:
norm1_13239549:
norm1_13239551:(
conv2_13239573:0
conv2_13239575:0
norm2_13239579:0
norm2_13239581:0
norm2_13239583:0
norm2_13239585:0"
dense1_13239614:	?@
dense1_13239616:@!
dense2_13239630:@@
dense2_13239632:@#
classify_13239654:@	
classify_13239656:	
identity?? classify/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?embed/StatefulPartitionedCall?norm1/StatefulPartitionedCall?norm2/StatefulPartitionedCall?
flatten_40/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_40_layer_call_and_return_conditional_losses_13239488?
embed/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0embed_13239501*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_embed_layer_call_and_return_conditional_losses_13239500?
reshape_20/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_13239518?
dropout_80/PartitionedCallPartitionedCall#reshape_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239525?
conv1/StatefulPartitionedCallStatefulPartitionedCall#dropout_80/PartitionedCall:output:0conv1_13239539conv1_13239541*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_13239538?
$average_pooling2d_40/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13239332?
norm1/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0norm1_13239545norm1_13239547norm1_13239549norm1_13239551*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm1_layer_call_and_return_conditional_losses_13239357?
dropout_81/PartitionedCallPartitionedCall&norm1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239559?
conv2/StatefulPartitionedCallStatefulPartitionedCall#dropout_81/PartitionedCall:output:0conv2_13239573conv2_13239575*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_13239572?
$average_pooling2d_41/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13239408?
norm2/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0norm2_13239579norm2_13239581norm2_13239583norm2_13239585*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm2_layer_call_and_return_conditional_losses_13239433?
flatten_41/PartitionedCallPartitionedCall&norm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_41_layer_call_and_return_conditional_losses_13239594?
dropout_82/PartitionedCallPartitionedCall#flatten_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239601?
dense1/StatefulPartitionedCallStatefulPartitionedCall#dropout_82/PartitionedCall:output:0dense1_13239614dense1_13239616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_13239613?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_13239630dense2_13239632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_13239629?
dropout_83/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239640?
 classify/StatefulPartitionedCallStatefulPartitionedCall#dropout_83/PartitionedCall:output:0classify_13239654classify_13239656*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_classify_layer_call_and_return_conditional_losses_13239653x
IdentityIdentity)classify/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp!^classify/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^embed/StatefulPartitionedCall^norm1/StatefulPartitionedCall^norm2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2D
 classify/StatefulPartitionedCall classify/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall2>
norm1/StatefulPartitionedCallnorm1/StatefulPartitionedCall2>
norm2/StatefulPartitionedCallnorm2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_flatten_40_layer_call_and_return_conditional_losses_13239488

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????
Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_dropout_83_layer_call_fn_13240888

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239640`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_dropout_81_layer_call_fn_13240693

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239559h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 
:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?
?
&__inference_cnn_layer_call_fn_13240057
flatten_40_input
unknown:	?J #
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:#
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:	?@

unknown_13:@

unknown_14:@@

unknown_15:@

unknown_16:@	

unknown_17:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_layer_call_and_return_conditional_losses_13239973o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????@
*
_user_specified_nameflatten_40_input
?
?
C__inference_norm2_layer_call_and_return_conditional_losses_13240789

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
D__inference_dense2_layer_call_and_return_conditional_losses_13239629

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_reshape_20_layer_call_and_return_conditional_losses_13239518

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@ `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
 :T P
,
_output_shapes
:??????????
 
 
_user_specified_nameinputs
?o
?
!__inference__traced_save_13241127
file_prefix/
+savev2_embed_embeddings_read_readvariableop+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop*
&savev2_norm1_gamma_read_readvariableop)
%savev2_norm1_beta_read_readvariableop0
,savev2_norm1_moving_mean_read_readvariableop4
0savev2_norm1_moving_variance_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop*
&savev2_norm2_gamma_read_readvariableop)
%savev2_norm2_beta_read_readvariableop0
,savev2_norm2_moving_mean_read_readvariableop4
0savev2_norm2_moving_variance_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop.
*savev2_classify_kernel_read_readvariableop,
(savev2_classify_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_embed_embeddings_m_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop1
-savev2_adam_norm1_gamma_m_read_readvariableop0
,savev2_adam_norm1_beta_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop1
-savev2_adam_norm2_gamma_m_read_readvariableop0
,savev2_adam_norm2_beta_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop5
1savev2_adam_classify_kernel_m_read_readvariableop3
/savev2_adam_classify_bias_m_read_readvariableop6
2savev2_adam_embed_embeddings_v_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop1
-savev2_adam_norm1_gamma_v_read_readvariableop0
,savev2_adam_norm1_beta_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop1
-savev2_adam_norm2_gamma_v_read_readvariableop0
,savev2_adam_norm2_beta_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop5
1savev2_adam_classify_kernel_v_read_readvariableop3
/savev2_adam_classify_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_embed_embeddings_read_readvariableop'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop&savev2_norm1_gamma_read_readvariableop%savev2_norm1_beta_read_readvariableop,savev2_norm1_moving_mean_read_readvariableop0savev2_norm1_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop&savev2_norm2_gamma_read_readvariableop%savev2_norm2_beta_read_readvariableop,savev2_norm2_moving_mean_read_readvariableop0savev2_norm2_moving_variance_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop*savev2_classify_kernel_read_readvariableop(savev2_classify_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_embed_embeddings_m_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop-savev2_adam_norm1_gamma_m_read_readvariableop,savev2_adam_norm1_beta_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop-savev2_adam_norm2_gamma_m_read_readvariableop,savev2_adam_norm2_beta_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop1savev2_adam_classify_kernel_m_read_readvariableop/savev2_adam_classify_bias_m_read_readvariableop2savev2_adam_embed_embeddings_v_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop-savev2_adam_norm1_gamma_v_read_readvariableop,savev2_adam_norm1_beta_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop-savev2_adam_norm2_gamma_v_read_readvariableop,savev2_adam_norm2_beta_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop1savev2_adam_classify_kernel_v_read_readvariableop/savev2_adam_classify_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?J : ::::::0:0:0:0:0:0:	?@:@:@@:@:@	:	: : : : : : : : : :	?J : ::::0:0:0:0:	?@:@:@@:@:@	:	:	?J : ::::0:0:0:0:	?@:@:@@:@:@	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?J :,(
&
_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:0: 	

_output_shapes
:0: 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?J :,(
&
_output_shapes
: : 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:0: #

_output_shapes
:0: $

_output_shapes
:0: %

_output_shapes
:0:%&!

_output_shapes
:	?@: '

_output_shapes
:@:$( 

_output_shapes

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@	: +

_output_shapes
:	:%,!

_output_shapes
:	?J :,-(
&
_output_shapes
: : .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::,1(
&
_output_shapes
:0: 2

_output_shapes
:0: 3

_output_shapes
:0: 4

_output_shapes
:0:%5!

_output_shapes
:	?@: 6

_output_shapes
:@:$7 

_output_shapes

:@@: 8

_output_shapes
:@:$9 

_output_shapes

:@	: :

_output_shapes
:	:;

_output_shapes
: 
?

?
F__inference_classify_layer_call_and_return_conditional_losses_13239653

inputs0
matmul_readvariableop_resource:@	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_reshape_20_layer_call_fn_13240555

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_13239518h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
 :T P
,
_output_shapes
:??????????
 
 
_user_specified_nameinputs
?
f
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240898

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_norm2_layer_call_and_return_conditional_losses_13240807

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
I
-__inference_dropout_82_layer_call_fn_13240823

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239601a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_embed_layer_call_and_return_conditional_losses_13239500

inputs,
embedding_lookup_13239494:	?J 
identity??embedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????
?
embedding_lookupResourceGatherembedding_lookup_13239494Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/13239494*,
_output_shapes
:??????????
 *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/13239494*,
_output_shapes
:??????????
 ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????
 x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????
 Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????
: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
(__inference_norm2_layer_call_fn_13240758

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm2_layer_call_and_return_conditional_losses_13239433?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
f
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239525

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@ :W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?
f
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239559

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 
:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?

g
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239774

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2_layer_call_and_return_conditional_losses_13240735

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 
0i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 
0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? 
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?
f
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240833

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_81_layer_call_fn_13240698

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239813w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?
?
)__inference_dense2_layer_call_fn_13240873

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_13239629o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

g
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239846

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@ *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@ :W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?	
?
D__inference_dense1_layer_call_and_return_conditional_losses_13239613

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense2_layer_call_and_return_conditional_losses_13240883

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
&__inference_cnn_layer_call_fn_13240269

inputs
unknown:	?J #
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:#
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:	?@

unknown_13:@

unknown_14:@@

unknown_15:@

unknown_16:@	

unknown_17:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_layer_call_and_return_conditional_losses_13239660o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_conv2_layer_call_fn_13240724

inputs!
unknown:0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_13239572w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 
0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? 
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?g
?
#__inference__wrapped_model_13239323
flatten_40_input6
#cnn_embed_embedding_lookup_13239238:	?J B
(cnn_conv1_conv2d_readvariableop_resource: 7
)cnn_conv1_biasadd_readvariableop_resource:/
!cnn_norm1_readvariableop_resource:1
#cnn_norm1_readvariableop_1_resource:@
2cnn_norm1_fusedbatchnormv3_readvariableop_resource:B
4cnn_norm1_fusedbatchnormv3_readvariableop_1_resource:B
(cnn_conv2_conv2d_readvariableop_resource:07
)cnn_conv2_biasadd_readvariableop_resource:0/
!cnn_norm2_readvariableop_resource:01
#cnn_norm2_readvariableop_1_resource:0@
2cnn_norm2_fusedbatchnormv3_readvariableop_resource:0B
4cnn_norm2_fusedbatchnormv3_readvariableop_1_resource:0<
)cnn_dense1_matmul_readvariableop_resource:	?@8
*cnn_dense1_biasadd_readvariableop_resource:@;
)cnn_dense2_matmul_readvariableop_resource:@@8
*cnn_dense2_biasadd_readvariableop_resource:@=
+cnn_classify_matmul_readvariableop_resource:@	:
,cnn_classify_biasadd_readvariableop_resource:	
identity??#cnn/classify/BiasAdd/ReadVariableOp?"cnn/classify/MatMul/ReadVariableOp? cnn/conv1/BiasAdd/ReadVariableOp?cnn/conv1/Conv2D/ReadVariableOp? cnn/conv2/BiasAdd/ReadVariableOp?cnn/conv2/Conv2D/ReadVariableOp?!cnn/dense1/BiasAdd/ReadVariableOp? cnn/dense1/MatMul/ReadVariableOp?!cnn/dense2/BiasAdd/ReadVariableOp? cnn/dense2/MatMul/ReadVariableOp?cnn/embed/embedding_lookup?)cnn/norm1/FusedBatchNormV3/ReadVariableOp?+cnn/norm1/FusedBatchNormV3/ReadVariableOp_1?cnn/norm1/ReadVariableOp?cnn/norm1/ReadVariableOp_1?)cnn/norm2/FusedBatchNormV3/ReadVariableOp?+cnn/norm2/FusedBatchNormV3/ReadVariableOp_1?cnn/norm2/ReadVariableOp?cnn/norm2/ReadVariableOp_1e
cnn/flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
cnn/flatten_40/ReshapeReshapeflatten_40_inputcnn/flatten_40/Const:output:0*
T0*(
_output_shapes
:??????????
y
cnn/embed/CastCastcnn/flatten_40/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????
?
cnn/embed/embedding_lookupResourceGather#cnn_embed_embedding_lookup_13239238cnn/embed/Cast:y:0*
Tindices0*6
_class,
*(loc:@cnn/embed/embedding_lookup/13239238*,
_output_shapes
:??????????
 *
dtype0?
#cnn/embed/embedding_lookup/IdentityIdentity#cnn/embed/embedding_lookup:output:0*
T0*6
_class,
*(loc:@cnn/embed/embedding_lookup/13239238*,
_output_shapes
:??????????
 ?
%cnn/embed/embedding_lookup/Identity_1Identity,cnn/embed/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????
 r
cnn/reshape_20/ShapeShape.cnn/embed/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:l
"cnn/reshape_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$cnn/reshape_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$cnn/reshape_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cnn/reshape_20/strided_sliceStridedSlicecnn/reshape_20/Shape:output:0+cnn/reshape_20/strided_slice/stack:output:0-cnn/reshape_20/strided_slice/stack_1:output:0-cnn/reshape_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
cnn/reshape_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@`
cnn/reshape_20/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`
cnn/reshape_20/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cnn/reshape_20/Reshape/shapePack%cnn/reshape_20/strided_slice:output:0'cnn/reshape_20/Reshape/shape/1:output:0'cnn/reshape_20/Reshape/shape/2:output:0'cnn/reshape_20/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
cnn/reshape_20/ReshapeReshape.cnn/embed/embedding_lookup/Identity_1:output:0%cnn/reshape_20/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@ ~
cnn/dropout_80/IdentityIdentitycnn/reshape_20/Reshape:output:0*
T0*/
_output_shapes
:?????????@ ?
cnn/conv1/Conv2D/ReadVariableOpReadVariableOp(cnn_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
cnn/conv1/Conv2DConv2D cnn/dropout_80/Identity:output:0'cnn/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 cnn/conv1/BiasAdd/ReadVariableOpReadVariableOp)cnn_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn/conv1/BiasAddBiasAddcnn/conv1/Conv2D:output:0(cnn/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
cnn/conv1/ReluRelucnn/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
 cnn/average_pooling2d_40/AvgPoolAvgPoolcnn/conv1/Relu:activations:0*
T0*/
_output_shapes
:????????? 
*
ksize
*
paddingVALID*
strides
v
cnn/norm1/ReadVariableOpReadVariableOp!cnn_norm1_readvariableop_resource*
_output_shapes
:*
dtype0z
cnn/norm1/ReadVariableOp_1ReadVariableOp#cnn_norm1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
)cnn/norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp2cnn_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
+cnn/norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4cnn_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
cnn/norm1/FusedBatchNormV3FusedBatchNormV3)cnn/average_pooling2d_40/AvgPool:output:0 cnn/norm1/ReadVariableOp:value:0"cnn/norm1/ReadVariableOp_1:value:01cnn/norm1/FusedBatchNormV3/ReadVariableOp:value:03cnn/norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? 
:::::*
epsilon%o?:*
is_training( }
cnn/dropout_81/IdentityIdentitycnn/norm1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 
?
cnn/conv2/Conv2D/ReadVariableOpReadVariableOp(cnn_conv2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
cnn/conv2/Conv2DConv2D cnn/dropout_81/Identity:output:0'cnn/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0*
paddingSAME*
strides
?
 cnn/conv2/BiasAdd/ReadVariableOpReadVariableOp)cnn_conv2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
cnn/conv2/BiasAddBiasAddcnn/conv2/Conv2D:output:0(cnn/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0l
cnn/conv2/ReluRelucnn/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
0?
 cnn/average_pooling2d_41/AvgPoolAvgPoolcnn/conv2/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
v
cnn/norm2/ReadVariableOpReadVariableOp!cnn_norm2_readvariableop_resource*
_output_shapes
:0*
dtype0z
cnn/norm2/ReadVariableOp_1ReadVariableOp#cnn_norm2_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
)cnn/norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp2cnn_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
+cnn/norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4cnn_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
cnn/norm2/FusedBatchNormV3FusedBatchNormV3)cnn/average_pooling2d_41/AvgPool:output:0 cnn/norm2/ReadVariableOp:value:0"cnn/norm2/ReadVariableOp_1:value:01cnn/norm2/FusedBatchNormV3/ReadVariableOp:value:03cnn/norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
is_training( e
cnn/flatten_41/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
cnn/flatten_41/ReshapeReshapecnn/norm2/FusedBatchNormV3:y:0cnn/flatten_41/Const:output:0*
T0*(
_output_shapes
:??????????w
cnn/dropout_82/IdentityIdentitycnn/flatten_41/Reshape:output:0*
T0*(
_output_shapes
:???????????
 cnn/dense1/MatMul/ReadVariableOpReadVariableOp)cnn_dense1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
cnn/dense1/MatMulMatMul cnn/dropout_82/Identity:output:0(cnn/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!cnn/dense1/BiasAdd/ReadVariableOpReadVariableOp*cnn_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn/dense1/BiasAddBiasAddcnn/dense1/MatMul:product:0)cnn/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 cnn/dense2/MatMul/ReadVariableOpReadVariableOp)cnn_dense2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
cnn/dense2/MatMulMatMulcnn/dense1/BiasAdd:output:0(cnn/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!cnn/dense2/BiasAdd/ReadVariableOpReadVariableOp*cnn_dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn/dense2/BiasAddBiasAddcnn/dense2/MatMul:product:0)cnn/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
cnn/dropout_83/IdentityIdentitycnn/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
"cnn/classify/MatMul/ReadVariableOpReadVariableOp+cnn_classify_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0?
cnn/classify/MatMulMatMul cnn/dropout_83/Identity:output:0*cnn/classify/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
#cnn/classify/BiasAdd/ReadVariableOpReadVariableOp,cnn_classify_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
cnn/classify/BiasAddBiasAddcnn/classify/MatMul:product:0+cnn/classify/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	p
cnn/classify/SoftmaxSoftmaxcnn/classify/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	m
IdentityIdentitycnn/classify/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp$^cnn/classify/BiasAdd/ReadVariableOp#^cnn/classify/MatMul/ReadVariableOp!^cnn/conv1/BiasAdd/ReadVariableOp ^cnn/conv1/Conv2D/ReadVariableOp!^cnn/conv2/BiasAdd/ReadVariableOp ^cnn/conv2/Conv2D/ReadVariableOp"^cnn/dense1/BiasAdd/ReadVariableOp!^cnn/dense1/MatMul/ReadVariableOp"^cnn/dense2/BiasAdd/ReadVariableOp!^cnn/dense2/MatMul/ReadVariableOp^cnn/embed/embedding_lookup*^cnn/norm1/FusedBatchNormV3/ReadVariableOp,^cnn/norm1/FusedBatchNormV3/ReadVariableOp_1^cnn/norm1/ReadVariableOp^cnn/norm1/ReadVariableOp_1*^cnn/norm2/FusedBatchNormV3/ReadVariableOp,^cnn/norm2/FusedBatchNormV3/ReadVariableOp_1^cnn/norm2/ReadVariableOp^cnn/norm2/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2J
#cnn/classify/BiasAdd/ReadVariableOp#cnn/classify/BiasAdd/ReadVariableOp2H
"cnn/classify/MatMul/ReadVariableOp"cnn/classify/MatMul/ReadVariableOp2D
 cnn/conv1/BiasAdd/ReadVariableOp cnn/conv1/BiasAdd/ReadVariableOp2B
cnn/conv1/Conv2D/ReadVariableOpcnn/conv1/Conv2D/ReadVariableOp2D
 cnn/conv2/BiasAdd/ReadVariableOp cnn/conv2/BiasAdd/ReadVariableOp2B
cnn/conv2/Conv2D/ReadVariableOpcnn/conv2/Conv2D/ReadVariableOp2F
!cnn/dense1/BiasAdd/ReadVariableOp!cnn/dense1/BiasAdd/ReadVariableOp2D
 cnn/dense1/MatMul/ReadVariableOp cnn/dense1/MatMul/ReadVariableOp2F
!cnn/dense2/BiasAdd/ReadVariableOp!cnn/dense2/BiasAdd/ReadVariableOp2D
 cnn/dense2/MatMul/ReadVariableOp cnn/dense2/MatMul/ReadVariableOp28
cnn/embed/embedding_lookupcnn/embed/embedding_lookup2V
)cnn/norm1/FusedBatchNormV3/ReadVariableOp)cnn/norm1/FusedBatchNormV3/ReadVariableOp2Z
+cnn/norm1/FusedBatchNormV3/ReadVariableOp_1+cnn/norm1/FusedBatchNormV3/ReadVariableOp_124
cnn/norm1/ReadVariableOpcnn/norm1/ReadVariableOp28
cnn/norm1/ReadVariableOp_1cnn/norm1/ReadVariableOp_12V
)cnn/norm2/FusedBatchNormV3/ReadVariableOp)cnn/norm2/FusedBatchNormV3/ReadVariableOp2Z
+cnn/norm2/FusedBatchNormV3/ReadVariableOp_1+cnn/norm2/FusedBatchNormV3/ReadVariableOp_124
cnn/norm2/ReadVariableOpcnn/norm2/ReadVariableOp28
cnn/norm2/ReadVariableOp_1cnn/norm2/ReadVariableOp_1:] Y
+
_output_shapes
:?????????@
*
_user_specified_nameflatten_40_input
?
?
C__inference_conv1_layer_call_and_return_conditional_losses_13239538

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?D
?
A__inference_cnn_layer_call_and_return_conditional_losses_13240116
flatten_40_input!
embed_13240061:	?J (
conv1_13240066: 
conv1_13240068:
norm1_13240072:
norm1_13240074:
norm1_13240076:
norm1_13240078:(
conv2_13240082:0
conv2_13240084:0
norm2_13240088:0
norm2_13240090:0
norm2_13240092:0
norm2_13240094:0"
dense1_13240099:	?@
dense1_13240101:@!
dense2_13240104:@@
dense2_13240106:@#
classify_13240110:@	
classify_13240112:	
identity?? classify/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?embed/StatefulPartitionedCall?norm1/StatefulPartitionedCall?norm2/StatefulPartitionedCall?
flatten_40/PartitionedCallPartitionedCallflatten_40_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_40_layer_call_and_return_conditional_losses_13239488?
embed/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0embed_13240061*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_embed_layer_call_and_return_conditional_losses_13239500?
reshape_20/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_13239518?
dropout_80/PartitionedCallPartitionedCall#reshape_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239525?
conv1/StatefulPartitionedCallStatefulPartitionedCall#dropout_80/PartitionedCall:output:0conv1_13240066conv1_13240068*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_13239538?
$average_pooling2d_40/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13239332?
norm1/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0norm1_13240072norm1_13240074norm1_13240076norm1_13240078*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm1_layer_call_and_return_conditional_losses_13239357?
dropout_81/PartitionedCallPartitionedCall&norm1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239559?
conv2/StatefulPartitionedCallStatefulPartitionedCall#dropout_81/PartitionedCall:output:0conv2_13240082conv2_13240084*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? 
0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_13239572?
$average_pooling2d_41/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13239408?
norm2/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0norm2_13240088norm2_13240090norm2_13240092norm2_13240094*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm2_layer_call_and_return_conditional_losses_13239433?
flatten_41/PartitionedCallPartitionedCall&norm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_41_layer_call_and_return_conditional_losses_13239594?
dropout_82/PartitionedCallPartitionedCall#flatten_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239601?
dense1/StatefulPartitionedCallStatefulPartitionedCall#dropout_82/PartitionedCall:output:0dense1_13240099dense1_13240101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_13239613?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_13240104dense2_13240106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_13239629?
dropout_83/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239640?
 classify/StatefulPartitionedCallStatefulPartitionedCall#dropout_83/PartitionedCall:output:0classify_13240110classify_13240112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_classify_layer_call_and_return_conditional_losses_13239653x
IdentityIdentity)classify/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp!^classify/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^embed/StatefulPartitionedCall^norm1/StatefulPartitionedCall^norm2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2D
 classify/StatefulPartitionedCall classify/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall2>
norm1/StatefulPartitionedCallnorm1/StatefulPartitionedCall2>
norm2/StatefulPartitionedCallnorm2/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????@
*
_user_specified_nameflatten_40_input
?
I
-__inference_flatten_40_layer_call_fn_13240527

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_40_layer_call_and_return_conditional_losses_13239488a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_reshape_20_layer_call_and_return_conditional_losses_13240569

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@ `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
 :T P
,
_output_shapes
:??????????
 
 
_user_specified_nameinputs
?
f
-__inference_dropout_80_layer_call_fn_13240579

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_80_layer_call_and_return_conditional_losses_13239846w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?
}
(__inference_embed_layer_call_fn_13240540

inputs
unknown:	?J 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_embed_layer_call_and_return_conditional_losses_13239500t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????
: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?

g
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240910

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239640

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
D__inference_dense1_layer_call_and_return_conditional_losses_13240864

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240584

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@ :W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?
d
H__inference_flatten_41_layer_call_and_return_conditional_losses_13239594

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
f
H__inference_dropout_82_layer_call_and_return_conditional_losses_13239601

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?"
$__inference__traced_restore_13241311
file_prefix4
!assignvariableop_embed_embeddings:	?J 9
assignvariableop_1_conv1_kernel: +
assignvariableop_2_conv1_bias:,
assignvariableop_3_norm1_gamma:+
assignvariableop_4_norm1_beta:2
$assignvariableop_5_norm1_moving_mean:6
(assignvariableop_6_norm1_moving_variance:9
assignvariableop_7_conv2_kernel:0+
assignvariableop_8_conv2_bias:0,
assignvariableop_9_norm2_gamma:0,
assignvariableop_10_norm2_beta:03
%assignvariableop_11_norm2_moving_mean:07
)assignvariableop_12_norm2_moving_variance:04
!assignvariableop_13_dense1_kernel:	?@-
assignvariableop_14_dense1_bias:@3
!assignvariableop_15_dense2_kernel:@@-
assignvariableop_16_dense2_bias:@5
#assignvariableop_17_classify_kernel:@	/
!assignvariableop_18_classify_bias:	'
assignvariableop_19_adam_iter:	 )
assignvariableop_20_adam_beta_1: )
assignvariableop_21_adam_beta_2: (
assignvariableop_22_adam_decay: 0
&assignvariableop_23_adam_learning_rate: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: #
assignvariableop_26_total: #
assignvariableop_27_count: >
+assignvariableop_28_adam_embed_embeddings_m:	?J A
'assignvariableop_29_adam_conv1_kernel_m: 3
%assignvariableop_30_adam_conv1_bias_m:4
&assignvariableop_31_adam_norm1_gamma_m:3
%assignvariableop_32_adam_norm1_beta_m:A
'assignvariableop_33_adam_conv2_kernel_m:03
%assignvariableop_34_adam_conv2_bias_m:04
&assignvariableop_35_adam_norm2_gamma_m:03
%assignvariableop_36_adam_norm2_beta_m:0;
(assignvariableop_37_adam_dense1_kernel_m:	?@4
&assignvariableop_38_adam_dense1_bias_m:@:
(assignvariableop_39_adam_dense2_kernel_m:@@4
&assignvariableop_40_adam_dense2_bias_m:@<
*assignvariableop_41_adam_classify_kernel_m:@	6
(assignvariableop_42_adam_classify_bias_m:	>
+assignvariableop_43_adam_embed_embeddings_v:	?J A
'assignvariableop_44_adam_conv1_kernel_v: 3
%assignvariableop_45_adam_conv1_bias_v:4
&assignvariableop_46_adam_norm1_gamma_v:3
%assignvariableop_47_adam_norm1_beta_v:A
'assignvariableop_48_adam_conv2_kernel_v:03
%assignvariableop_49_adam_conv2_bias_v:04
&assignvariableop_50_adam_norm2_gamma_v:03
%assignvariableop_51_adam_norm2_beta_v:0;
(assignvariableop_52_adam_dense1_kernel_v:	?@4
&assignvariableop_53_adam_dense1_bias_v:@:
(assignvariableop_54_adam_dense2_kernel_v:@@4
&assignvariableop_55_adam_dense2_bias_v:@<
*assignvariableop_56_adam_classify_kernel_v:@	6
(assignvariableop_57_adam_classify_bias_v:	
identity_59??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_embed_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_norm1_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_norm1_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_norm1_moving_meanIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_norm1_moving_varianceIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_norm2_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_norm2_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_norm2_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_norm2_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_dense1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_dense2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_classify_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_classify_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_iterIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_decayIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adam_learning_rateIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_embed_embeddings_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_conv1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_conv1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_adam_norm1_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_norm1_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_conv2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_conv2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp&assignvariableop_35_adam_norm2_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_norm2_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_dense1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_dense2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_classify_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_classify_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_embed_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_conv1_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_conv1_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_norm1_gamma_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_norm1_beta_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_conv2_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp%assignvariableop_49_adam_conv2_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_norm2_gamma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_norm2_beta_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense1_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp&assignvariableop_53_adam_dense1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense2_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp&assignvariableop_55_adam_dense2_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_classify_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_classify_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_59IdentityIdentity_58:output:0^NoOp_1*
T0*
_output_shapes
: ?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_59Identity_59:output:0*?
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
f
-__inference_dropout_83_layer_call_fn_13240893

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_83_layer_call_and_return_conditional_losses_13239731o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
n
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13239408

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

g
H__inference_dropout_81_layer_call_and_return_conditional_losses_13239813

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? 
*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 
:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_13240226
flatten_40_input
unknown:	?J #
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:#
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:	?@

unknown_13:@

unknown_14:@@

unknown_15:@

unknown_16:@	

unknown_17:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_13239323o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????@
*
_user_specified_nameflatten_40_input
?
?
C__inference_conv2_layer_call_and_return_conditional_losses_13239572

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 
0i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 
0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? 
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
??
?
A__inference_cnn_layer_call_and_return_conditional_losses_13240522

inputs2
embed_embedding_lookup_13240409:	?J >
$conv1_conv2d_readvariableop_resource: 3
%conv1_biasadd_readvariableop_resource:+
norm1_readvariableop_resource:-
norm1_readvariableop_1_resource:<
.norm1_fusedbatchnormv3_readvariableop_resource:>
0norm1_fusedbatchnormv3_readvariableop_1_resource:>
$conv2_conv2d_readvariableop_resource:03
%conv2_biasadd_readvariableop_resource:0+
norm2_readvariableop_resource:0-
norm2_readvariableop_1_resource:0<
.norm2_fusedbatchnormv3_readvariableop_resource:0>
0norm2_fusedbatchnormv3_readvariableop_1_resource:08
%dense1_matmul_readvariableop_resource:	?@4
&dense1_biasadd_readvariableop_resource:@7
%dense2_matmul_readvariableop_resource:@@4
&dense2_biasadd_readvariableop_resource:@9
'classify_matmul_readvariableop_resource:@	6
(classify_biasadd_readvariableop_resource:	
identity??classify/BiasAdd/ReadVariableOp?classify/MatMul/ReadVariableOp?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?embed/embedding_lookup?norm1/AssignNewValue?norm1/AssignNewValue_1?%norm1/FusedBatchNormV3/ReadVariableOp?'norm1/FusedBatchNormV3/ReadVariableOp_1?norm1/ReadVariableOp?norm1/ReadVariableOp_1?norm2/AssignNewValue?norm2/AssignNewValue_1?%norm2/FusedBatchNormV3/ReadVariableOp?'norm2/FusedBatchNormV3/ReadVariableOp_1?norm2/ReadVariableOp?norm2/ReadVariableOp_1a
flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   s
flatten_40/ReshapeReshapeinputsflatten_40/Const:output:0*
T0*(
_output_shapes
:??????????
q

embed/CastCastflatten_40/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????
?
embed/embedding_lookupResourceGatherembed_embedding_lookup_13240409embed/Cast:y:0*
Tindices0*2
_class(
&$loc:@embed/embedding_lookup/13240409*,
_output_shapes
:??????????
 *
dtype0?
embed/embedding_lookup/IdentityIdentityembed/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embed/embedding_lookup/13240409*,
_output_shapes
:??????????
 ?
!embed/embedding_lookup/Identity_1Identity(embed/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????
 j
reshape_20/ShapeShape*embed/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:h
reshape_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_20/strided_sliceStridedSlicereshape_20/Shape:output:0'reshape_20/strided_slice/stack:output:0)reshape_20/strided_slice/stack_1:output:0)reshape_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@\
reshape_20/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_20/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
reshape_20/Reshape/shapePack!reshape_20/strided_slice:output:0#reshape_20/Reshape/shape/1:output:0#reshape_20/Reshape/shape/2:output:0#reshape_20/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_20/ReshapeReshape*embed/embedding_lookup/Identity_1:output:0!reshape_20/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@ ]
dropout_80/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_80/dropout/MulMulreshape_20/Reshape:output:0!dropout_80/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@ c
dropout_80/dropout/ShapeShapereshape_20/Reshape:output:0*
T0*
_output_shapes
:?
/dropout_80/dropout/random_uniform/RandomUniformRandomUniform!dropout_80/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@ *
dtype0*

seed*f
!dropout_80/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_80/dropout/GreaterEqualGreaterEqual8dropout_80/dropout/random_uniform/RandomUniform:output:0*dropout_80/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@ ?
dropout_80/dropout/CastCast#dropout_80/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@ ?
dropout_80/dropout/Mul_1Muldropout_80/dropout/Mul:z:0dropout_80/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@ ?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv1/Conv2DConv2Ddropout_80/dropout/Mul_1:z:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
average_pooling2d_40/AvgPoolAvgPoolconv1/Relu:activations:0*
T0*/
_output_shapes
:????????? 
*
ksize
*
paddingVALID*
strides
n
norm1/ReadVariableOpReadVariableOpnorm1_readvariableop_resource*
_output_shapes
:*
dtype0r
norm1/ReadVariableOp_1ReadVariableOpnorm1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp.norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
'norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
norm1/FusedBatchNormV3FusedBatchNormV3%average_pooling2d_40/AvgPool:output:0norm1/ReadVariableOp:value:0norm1/ReadVariableOp_1:value:0-norm1/FusedBatchNormV3/ReadVariableOp:value:0/norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? 
:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
norm1/AssignNewValueAssignVariableOp.norm1_fusedbatchnormv3_readvariableop_resource#norm1/FusedBatchNormV3:batch_mean:0&^norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
norm1/AssignNewValue_1AssignVariableOp0norm1_fusedbatchnormv3_readvariableop_1_resource'norm1/FusedBatchNormV3:batch_variance:0(^norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(]
dropout_81/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_81/dropout/MulMulnorm1/FusedBatchNormV3:y:0!dropout_81/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 
b
dropout_81/dropout/ShapeShapenorm1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
/dropout_81/dropout/random_uniform/RandomUniformRandomUniform!dropout_81/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? 
*
dtype0*

seed**
seed2f
!dropout_81/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_81/dropout/GreaterEqualGreaterEqual8dropout_81/dropout/random_uniform/RandomUniform:output:0*dropout_81/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 
?
dropout_81/dropout/CastCast#dropout_81/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 
?
dropout_81/dropout/Mul_1Muldropout_81/dropout/Mul:z:0dropout_81/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 
?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2/Conv2DConv2Ddropout_81/dropout/Mul_1:z:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0*
paddingSAME*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
0d

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
0?
average_pooling2d_41/AvgPoolAvgPoolconv2/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
n
norm2/ReadVariableOpReadVariableOpnorm2_readvariableop_resource*
_output_shapes
:0*
dtype0r
norm2/ReadVariableOp_1ReadVariableOpnorm2_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
%norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp.norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
'norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
norm2/FusedBatchNormV3FusedBatchNormV3%average_pooling2d_41/AvgPool:output:0norm2/ReadVariableOp:value:0norm2/ReadVariableOp_1:value:0-norm2/FusedBatchNormV3/ReadVariableOp:value:0/norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
norm2/AssignNewValueAssignVariableOp.norm2_fusedbatchnormv3_readvariableop_resource#norm2/FusedBatchNormV3:batch_mean:0&^norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
norm2/AssignNewValue_1AssignVariableOp0norm2_fusedbatchnormv3_readvariableop_1_resource'norm2/FusedBatchNormV3:batch_variance:0(^norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(a
flatten_41/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_41/ReshapeReshapenorm2/FusedBatchNormV3:y:0flatten_41/Const:output:0*
T0*(
_output_shapes
:??????????]
dropout_82/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_82/dropout/MulMulflatten_41/Reshape:output:0!dropout_82/dropout/Const:output:0*
T0*(
_output_shapes
:??????????c
dropout_82/dropout/ShapeShapeflatten_41/Reshape:output:0*
T0*
_output_shapes
:?
/dropout_82/dropout/random_uniform/RandomUniformRandomUniform!dropout_82/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2f
!dropout_82/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_82/dropout/GreaterEqualGreaterEqual8dropout_82/dropout/random_uniform/RandomUniform:output:0*dropout_82/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_82/dropout/CastCast#dropout_82/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_82/dropout/Mul_1Muldropout_82/dropout/Mul:z:0dropout_82/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense1/MatMulMatMuldropout_82/dropout/Mul_1:z:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
dense2/MatMulMatMuldense1/BiasAdd:output:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@]
dropout_83/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_83/dropout/MulMuldense2/BiasAdd:output:0!dropout_83/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@_
dropout_83/dropout/ShapeShapedense2/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_83/dropout/random_uniform/RandomUniformRandomUniform!dropout_83/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0*

seed**
seed2f
!dropout_83/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_83/dropout/GreaterEqualGreaterEqual8dropout_83/dropout/random_uniform/RandomUniform:output:0*dropout_83/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
dropout_83/dropout/CastCast#dropout_83/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
dropout_83/dropout/Mul_1Muldropout_83/dropout/Mul:z:0dropout_83/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
classify/MatMul/ReadVariableOpReadVariableOp'classify_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0?
classify/MatMulMatMuldropout_83/dropout/Mul_1:z:0&classify/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
classify/BiasAdd/ReadVariableOpReadVariableOp(classify_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
classify/BiasAddBiasAddclassify/MatMul:product:0'classify/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	h
classify/SoftmaxSoftmaxclassify/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	i
IdentityIdentityclassify/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	?
NoOpNoOp ^classify/BiasAdd/ReadVariableOp^classify/MatMul/ReadVariableOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^embed/embedding_lookup^norm1/AssignNewValue^norm1/AssignNewValue_1&^norm1/FusedBatchNormV3/ReadVariableOp(^norm1/FusedBatchNormV3/ReadVariableOp_1^norm1/ReadVariableOp^norm1/ReadVariableOp_1^norm2/AssignNewValue^norm2/AssignNewValue_1&^norm2/FusedBatchNormV3/ReadVariableOp(^norm2/FusedBatchNormV3/ReadVariableOp_1^norm2/ReadVariableOp^norm2/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 2B
classify/BiasAdd/ReadVariableOpclassify/BiasAdd/ReadVariableOp2@
classify/MatMul/ReadVariableOpclassify/MatMul/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp20
embed/embedding_lookupembed/embedding_lookup2,
norm1/AssignNewValuenorm1/AssignNewValue20
norm1/AssignNewValue_1norm1/AssignNewValue_12N
%norm1/FusedBatchNormV3/ReadVariableOp%norm1/FusedBatchNormV3/ReadVariableOp2R
'norm1/FusedBatchNormV3/ReadVariableOp_1'norm1/FusedBatchNormV3/ReadVariableOp_12,
norm1/ReadVariableOpnorm1/ReadVariableOp20
norm1/ReadVariableOp_1norm1/ReadVariableOp_12,
norm2/AssignNewValuenorm2/AssignNewValue20
norm2/AssignNewValue_1norm2/AssignNewValue_12N
%norm2/FusedBatchNormV3/ReadVariableOp%norm2/FusedBatchNormV3/ReadVariableOp2R
'norm2/FusedBatchNormV3/ReadVariableOp_1'norm2/FusedBatchNormV3/ReadVariableOp_12,
norm2/ReadVariableOpnorm2/ReadVariableOp20
norm2/ReadVariableOp_1norm2/ReadVariableOp_1:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
S
7__inference_average_pooling2d_40_layer_call_fn_13240621

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13239332?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv1_layer_call_fn_13240605

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_13239538w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@ 
 
_user_specified_nameinputs
?

g
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240845

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_embed_layer_call_and_return_conditional_losses_13240550

inputs,
embedding_lookup_13240544:	?J 
identity??embedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????
?
embedding_lookupResourceGatherembedding_lookup_13240544Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/13240544*,
_output_shapes
:??????????
 *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/13240544*,
_output_shapes
:??????????
 ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????
 x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????
 Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????
: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
f
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240703

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 
:W S
/
_output_shapes
:????????? 

 
_user_specified_nameinputs
?
?
C__inference_norm1_layer_call_and_return_conditional_losses_13239357

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_cnn_layer_call_fn_13240312

inputs
unknown:	?J #
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:#
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:	?@

unknown_13:@

unknown_14:@@

unknown_15:@

unknown_16:@	

unknown_17:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_layer_call_and_return_conditional_losses_13239973o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????@: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
n
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13240745

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_norm1_layer_call_fn_13240639

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_norm1_layer_call_and_return_conditional_losses_13239357?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_classify_layer_call_and_return_conditional_losses_13240930

inputs0
matmul_readvariableop_resource:@	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
flatten_40_input=
"serving_default_flatten_40_input:0?????????@<
classify0
StatefulPartitionedCall:0?????????	tensorflow/serving/predict:á
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'
embeddings"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator"
_tf_keras_layer
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance"
_tf_keras_layer
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|_random_generator"
_tf_keras_layer
?
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
'0
;1
<2
K3
L4
M5
N6
\7
]8
l9
m10
n11
o12
?13
?14
?15
?16
?17
?18"
trackable_list_wrapper
?
;0
<1
K2
L3
\4
]5
l6
m7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
&__inference_cnn_layer_call_fn_13239701
&__inference_cnn_layer_call_fn_13240269
&__inference_cnn_layer_call_fn_13240312
&__inference_cnn_layer_call_fn_13240057?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
A__inference_cnn_layer_call_and_return_conditional_losses_13240403
A__inference_cnn_layer_call_and_return_conditional_losses_13240522
A__inference_cnn_layer_call_and_return_conditional_losses_13240116
A__inference_cnn_layer_call_and_return_conditional_losses_13240175?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
#__inference__wrapped_model_13239323flatten_40_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate'm?;m?<m?Km?Lm?\m?]m?lm?mm?	?m?	?m?	?m?	?m?	?m?	?m?'v?;v?<v?Kv?Lv?\v?]v?lv?mv?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_flatten_40_layer_call_fn_13240527?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_flatten_40_layer_call_and_return_conditional_losses_13240533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_embed_layer_call_fn_13240540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_embed_layer_call_and_return_conditional_losses_13240550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
#:!	?J 2embed/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_reshape_20_layer_call_fn_13240555?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_reshape_20_layer_call_and_return_conditional_losses_13240569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
-__inference_dropout_80_layer_call_fn_13240574
-__inference_dropout_80_layer_call_fn_13240579?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240584
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240596?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv1_layer_call_fn_13240605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv1_layer_call_and_return_conditional_losses_13240616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
&:$ 2conv1/kernel
:2
conv1/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
7__inference_average_pooling2d_40_layer_call_fn_13240621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13240626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
(__inference_norm1_layer_call_fn_13240639
(__inference_norm1_layer_call_fn_13240652?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
C__inference_norm1_layer_call_and_return_conditional_losses_13240670
C__inference_norm1_layer_call_and_return_conditional_losses_13240688?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
:2norm1/gamma
:2
norm1/beta
!: (2norm1/moving_mean
%:# (2norm1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
-__inference_dropout_81_layer_call_fn_13240693
-__inference_dropout_81_layer_call_fn_13240698?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240703
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240715?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2_layer_call_fn_13240724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2_layer_call_and_return_conditional_losses_13240735?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
&:$02conv2/kernel
:02
conv2/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
7__inference_average_pooling2d_41_layer_call_fn_13240740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13240745?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
l0
m1
n2
o3"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
(__inference_norm2_layer_call_fn_13240758
(__inference_norm2_layer_call_fn_13240771?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
C__inference_norm2_layer_call_and_return_conditional_losses_13240789
C__inference_norm2_layer_call_and_return_conditional_losses_13240807?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
:02norm2/gamma
:02
norm2/beta
!:0 (2norm2/moving_mean
%:#0 (2norm2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_flatten_41_layer_call_fn_13240812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_flatten_41_layer_call_and_return_conditional_losses_13240818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
-__inference_dropout_82_layer_call_fn_13240823
-__inference_dropout_82_layer_call_fn_13240828?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240833
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240845?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense1_layer_call_fn_13240854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_dense1_layer_call_and_return_conditional_losses_13240864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 :	?@2dense1/kernel
:@2dense1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense2_layer_call_fn_13240873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_dense2_layer_call_and_return_conditional_losses_13240883?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
:@@2dense2/kernel
:@2dense2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
-__inference_dropout_83_layer_call_fn_13240888
-__inference_dropout_83_layer_call_fn_13240893?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240898
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_classify_layer_call_fn_13240919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_classify_layer_call_and_return_conditional_losses_13240930?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:@	2classify/kernel
:	2classify/bias
C
'0
M1
N2
n3
o4"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_cnn_layer_call_fn_13239701flatten_40_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_cnn_layer_call_fn_13240269inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_cnn_layer_call_fn_13240312inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_cnn_layer_call_fn_13240057flatten_40_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_cnn_layer_call_and_return_conditional_losses_13240403inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_cnn_layer_call_and_return_conditional_losses_13240522inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_cnn_layer_call_and_return_conditional_losses_13240116flatten_40_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_cnn_layer_call_and_return_conditional_losses_13240175flatten_40_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
&__inference_signature_wrapper_13240226flatten_40_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
-__inference_flatten_40_layer_call_fn_13240527inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_flatten_40_layer_call_and_return_conditional_losses_13240533inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_embed_layer_call_fn_13240540inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_embed_layer_call_and_return_conditional_losses_13240550inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
-__inference_reshape_20_layer_call_fn_13240555inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_reshape_20_layer_call_and_return_conditional_losses_13240569inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
-__inference_dropout_80_layer_call_fn_13240574inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_dropout_80_layer_call_fn_13240579inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240584inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240596inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
(__inference_conv1_layer_call_fn_13240605inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv1_layer_call_and_return_conditional_losses_13240616inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
7__inference_average_pooling2d_40_layer_call_fn_13240621inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13240626inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_norm1_layer_call_fn_13240639inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_norm1_layer_call_fn_13240652inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_norm1_layer_call_and_return_conditional_losses_13240670inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_norm1_layer_call_and_return_conditional_losses_13240688inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
-__inference_dropout_81_layer_call_fn_13240693inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_dropout_81_layer_call_fn_13240698inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240703inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240715inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
(__inference_conv2_layer_call_fn_13240724inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2_layer_call_and_return_conditional_losses_13240735inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
7__inference_average_pooling2d_41_layer_call_fn_13240740inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13240745inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_norm2_layer_call_fn_13240758inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_norm2_layer_call_fn_13240771inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_norm2_layer_call_and_return_conditional_losses_13240789inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_norm2_layer_call_and_return_conditional_losses_13240807inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
-__inference_flatten_41_layer_call_fn_13240812inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_flatten_41_layer_call_and_return_conditional_losses_13240818inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
-__inference_dropout_82_layer_call_fn_13240823inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_dropout_82_layer_call_fn_13240828inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240833inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240845inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
)__inference_dense1_layer_call_fn_13240854inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense1_layer_call_and_return_conditional_losses_13240864inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dense2_layer_call_fn_13240873inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense2_layer_call_and_return_conditional_losses_13240883inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
-__inference_dropout_83_layer_call_fn_13240888inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_dropout_83_layer_call_fn_13240893inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240898inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240910inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
+__inference_classify_layer_call_fn_13240919inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_classify_layer_call_and_return_conditional_losses_13240930inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
(:&	?J 2Adam/embed/embeddings/m
+:) 2Adam/conv1/kernel/m
:2Adam/conv1/bias/m
:2Adam/norm1/gamma/m
:2Adam/norm1/beta/m
+:)02Adam/conv2/kernel/m
:02Adam/conv2/bias/m
:02Adam/norm2/gamma/m
:02Adam/norm2/beta/m
%:#	?@2Adam/dense1/kernel/m
:@2Adam/dense1/bias/m
$:"@@2Adam/dense2/kernel/m
:@2Adam/dense2/bias/m
&:$@	2Adam/classify/kernel/m
 :	2Adam/classify/bias/m
(:&	?J 2Adam/embed/embeddings/v
+:) 2Adam/conv1/kernel/v
:2Adam/conv1/bias/v
:2Adam/norm1/gamma/v
:2Adam/norm1/beta/v
+:)02Adam/conv2/kernel/v
:02Adam/conv2/bias/v
:02Adam/norm2/gamma/v
:02Adam/norm2/beta/v
%:#	?@2Adam/dense1/kernel/v
:@2Adam/dense1/bias/v
$:"@@2Adam/dense2/kernel/v
:@2Adam/dense2/bias/v
&:$@	2Adam/classify/kernel/v
 :	2Adam/classify/bias/v?
#__inference__wrapped_model_13239323?';<KLMN\]lmno??????=?:
3?0
.?+
flatten_40_input?????????@
? "3?0
.
classify"?
classify?????????	?
R__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_13240626?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
7__inference_average_pooling2d_40_layer_call_fn_13240621?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
R__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_13240745?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
7__inference_average_pooling2d_41_layer_call_fn_13240740?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_classify_layer_call_and_return_conditional_losses_13240930^??/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????	
? ?
+__inference_classify_layer_call_fn_13240919Q??/?,
%?"
 ?
inputs?????????@
? "??????????	?
A__inference_cnn_layer_call_and_return_conditional_losses_13240116?';<KLMN\]lmno??????E?B
;?8
.?+
flatten_40_input?????????@
p 

 
? "%?"
?
0?????????	
? ?
A__inference_cnn_layer_call_and_return_conditional_losses_13240175?';<KLMN\]lmno??????E?B
;?8
.?+
flatten_40_input?????????@
p

 
? "%?"
?
0?????????	
? ?
A__inference_cnn_layer_call_and_return_conditional_losses_13240403';<KLMN\]lmno??????;?8
1?.
$?!
inputs?????????@
p 

 
? "%?"
?
0?????????	
? ?
A__inference_cnn_layer_call_and_return_conditional_losses_13240522';<KLMN\]lmno??????;?8
1?.
$?!
inputs?????????@
p

 
? "%?"
?
0?????????	
? ?
&__inference_cnn_layer_call_fn_13239701|';<KLMN\]lmno??????E?B
;?8
.?+
flatten_40_input?????????@
p 

 
? "??????????	?
&__inference_cnn_layer_call_fn_13240057|';<KLMN\]lmno??????E?B
;?8
.?+
flatten_40_input?????????@
p

 
? "??????????	?
&__inference_cnn_layer_call_fn_13240269r';<KLMN\]lmno??????;?8
1?.
$?!
inputs?????????@
p 

 
? "??????????	?
&__inference_cnn_layer_call_fn_13240312r';<KLMN\]lmno??????;?8
1?.
$?!
inputs?????????@
p

 
? "??????????	?
C__inference_conv1_layer_call_and_return_conditional_losses_13240616l;<7?4
-?*
(?%
inputs?????????@ 
? "-?*
#? 
0?????????@
? ?
(__inference_conv1_layer_call_fn_13240605_;<7?4
-?*
(?%
inputs?????????@ 
? " ??????????@?
C__inference_conv2_layer_call_and_return_conditional_losses_13240735l\]7?4
-?*
(?%
inputs????????? 

? "-?*
#? 
0????????? 
0
? ?
(__inference_conv2_layer_call_fn_13240724_\]7?4
-?*
(?%
inputs????????? 

? " ?????????? 
0?
D__inference_dense1_layer_call_and_return_conditional_losses_13240864_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? 
)__inference_dense1_layer_call_fn_13240854R??0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dense2_layer_call_and_return_conditional_losses_13240883^??/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
)__inference_dense2_layer_call_fn_13240873Q??/?,
%?"
 ?
inputs?????????@
? "??????????@?
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240584l;?8
1?.
(?%
inputs?????????@ 
p 
? "-?*
#? 
0?????????@ 
? ?
H__inference_dropout_80_layer_call_and_return_conditional_losses_13240596l;?8
1?.
(?%
inputs?????????@ 
p
? "-?*
#? 
0?????????@ 
? ?
-__inference_dropout_80_layer_call_fn_13240574_;?8
1?.
(?%
inputs?????????@ 
p 
? " ??????????@ ?
-__inference_dropout_80_layer_call_fn_13240579_;?8
1?.
(?%
inputs?????????@ 
p
? " ??????????@ ?
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240703l;?8
1?.
(?%
inputs????????? 

p 
? "-?*
#? 
0????????? 

? ?
H__inference_dropout_81_layer_call_and_return_conditional_losses_13240715l;?8
1?.
(?%
inputs????????? 

p
? "-?*
#? 
0????????? 

? ?
-__inference_dropout_81_layer_call_fn_13240693_;?8
1?.
(?%
inputs????????? 

p 
? " ?????????? 
?
-__inference_dropout_81_layer_call_fn_13240698_;?8
1?.
(?%
inputs????????? 

p
? " ?????????? 
?
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240833^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
H__inference_dropout_82_layer_call_and_return_conditional_losses_13240845^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
-__inference_dropout_82_layer_call_fn_13240823Q4?1
*?'
!?
inputs??????????
p 
? "????????????
-__inference_dropout_82_layer_call_fn_13240828Q4?1
*?'
!?
inputs??????????
p
? "????????????
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240898\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
H__inference_dropout_83_layer_call_and_return_conditional_losses_13240910\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
-__inference_dropout_83_layer_call_fn_13240888O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
-__inference_dropout_83_layer_call_fn_13240893O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
C__inference_embed_layer_call_and_return_conditional_losses_13240550a'0?-
&?#
!?
inputs??????????

? "*?'
 ?
0??????????
 
? ?
(__inference_embed_layer_call_fn_13240540T'0?-
&?#
!?
inputs??????????

? "???????????
 ?
H__inference_flatten_40_layer_call_and_return_conditional_losses_13240533]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????

? ?
-__inference_flatten_40_layer_call_fn_13240527P3?0
)?&
$?!
inputs?????????@
? "???????????
?
H__inference_flatten_41_layer_call_and_return_conditional_losses_13240818a7?4
-?*
(?%
inputs?????????0
? "&?#
?
0??????????
? ?
-__inference_flatten_41_layer_call_fn_13240812T7?4
-?*
(?%
inputs?????????0
? "????????????
C__inference_norm1_layer_call_and_return_conditional_losses_13240670?KLMNM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_norm1_layer_call_and_return_conditional_losses_13240688?KLMNM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
(__inference_norm1_layer_call_fn_13240639?KLMNM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
(__inference_norm1_layer_call_fn_13240652?KLMNM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
C__inference_norm2_layer_call_and_return_conditional_losses_13240789?lmnoM?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
C__inference_norm2_layer_call_and_return_conditional_losses_13240807?lmnoM?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
(__inference_norm2_layer_call_fn_13240758?lmnoM?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
(__inference_norm2_layer_call_fn_13240771?lmnoM?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
H__inference_reshape_20_layer_call_and_return_conditional_losses_13240569e4?1
*?'
%?"
inputs??????????
 
? "-?*
#? 
0?????????@ 
? ?
-__inference_reshape_20_layer_call_fn_13240555X4?1
*?'
%?"
inputs??????????
 
? " ??????????@ ?
&__inference_signature_wrapper_13240226?';<KLMN\]lmno??????Q?N
? 
G?D
B
flatten_40_input.?+
flatten_40_input?????????@"3?0
.
classify"?
classify?????????	
�	{�"0և8@{�"0և8@!{�"0և8@	겸���?겸���?!겸���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails${�"0և8@�p�q�t�?A>�D��8@YG�0}�!�?*	��C�l�k@2U
Iterator::Model::ParallelMapV2{2��4�?!�*�A�>@){2��4�?1�*�A�>@:Preprocessing2F
Iterator::Model� ��F!�?!����qI@)��KqU٧?1@�~���4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0��B�?!Ȥ�Q�~7@)�I��Gp�?1"���0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���E��?!Aٲ�A�3@)A��4F�?1��U�j�-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceo���׍?!�Vb�1@)o���׍?1�Vb�1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ�i�W�?!_�1�@)J�i�W�?1_�1�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��N�?!zd�W�H@)���C�r�?1=��Y�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy�\�?!�N�Wi�8@)OYM�]g?1С�`@h�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9겸���?I��,6��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�p�q�t�?�p�q�t�?!�p�q�t�?      ��!       "      ��!       *      ��!       2	>�D��8@>�D��8@!>�D��8@:      ��!       B      ��!       J	G�0}�!�?G�0}�!�?!G�0}�!�?R      ��!       Z	G�0}�!�?G�0}�!�?!G�0}�!�?b      ��!       JCPU_ONLYY겸���?b q��,6��X@Y      Y@q!^l;_D@"�	
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�40.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 
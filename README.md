# hwr
Handwriting Recognition 2018 project

# to run CNN, use the following command in terminal in CNN_test file directory
python retrain.py --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=500 --model_dir=inception --summaries_dir=tf_files/training_summaries/basic --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --image_dir=../monkbrill_jpg

# to run CNN with augmentation, use the following command in terminal in CNN_test file directory
python retrain.py --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=750 --model_dir=inception --summaries_dir=tf_files/training_summaries/basic --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --image_dir=../monkbrill_aug


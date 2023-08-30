start_time=$(date +%s)

echo "starting the medical robot..."
echo "currently in env med_robot"

python answer.py & # env med_robot
python prompt/ready.py
sleep 5
python prompt/build_index.py &

# wait for 20 secs to build indexing
sleep 12

python prompt/record_audio.py
conda activate stt
echo "currently in env stt"
python record_audio.py # env stt

python stt.py # env stt

# wait 5 secs for answering
sleep 5

conda activate tts
echo "currently in env tts"
python prompt/analyze.py 

python prompt/load_database.py & 
python prompt/generate_answer.py &

python tts.py  # env tts

conda activate med_robot
echo "currently in env med_robot"

end_time=$(date +%s)
echo "total time spent in this query-answer: $((end_time-start_time)) secs"

python prompt/end.py &

rm ./conversation/*
rm ./wav/*
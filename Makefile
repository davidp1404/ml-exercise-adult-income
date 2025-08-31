experiments:
	dvc exp run --queue -S train.n_estimators=100 -S train.max_depth=20 -m "n_estimators and max_depth"
	dvc exp run --queue -S train.max_depth=20 -m "max_depth only"
	dvc exp show

run:
	dvc queue start --jobs 2
	dvc exp show

clean:
	dvc exp remove --queue -A
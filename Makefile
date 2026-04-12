PROCESSED_DIR = data/processed
PRE_DIR = data/preprocessed
SRC_DIR = src

.PHONY: prepare preprocess run-lg run-gb run-rf compare clean

prepare:
	@mkdir -p $(PROCESSED_DIR) $(PRE_DIR)

preprocess: prepare
	@echo "Running Preprocessing..."
	python3 $(SRC_DIR)/preprocessing.py

run-lg: prepare
	@echo "Training Logistic Regression..."
	python3 $(SRC_DIR)/train_logistic_regression.py
	@V_NUM=$$(ls $(PROCESSED_DIR)/lg-v*.csv 2>/dev/null | wc -l | tr -d ' ' | awk '{print $$1 + 1}'); \
	if [ -f $(PROCESSED_DIR)/submission_lg.csv ]; then \
		mv $(PROCESSED_DIR)/submission_lg.csv $(PROCESSED_DIR)/lg-v$$V_NUM.csv; \
		echo "Success: Created $(PROCESSED_DIR)/lg-v$$V_NUM.csv"; \
	fi

run-gb: prepare
	@echo "Training Gradient Boosting..."
	python3 $(SRC_DIR)/train_gradient_boosting.py
	@V_NUM=$$(ls $(PROCESSED_DIR)/gb-v*.csv 2>/dev/null | wc -l | tr -d ' ' | awk '{print $$1 + 1}'); \
	if [ -f $(PROCESSED_DIR)/submission_gb.csv ]; then \
		mv $(PROCESSED_DIR)/submission_gb.csv $(PROCESSED_DIR)/gb-v$$V_NUM.csv; \
		echo "Success: Created $(PROCESSED_DIR)/gb-v$$V_NUM.csv"; \
	fi

run-rf: prepare
	@echo "Training Random Forest..."
	python3 $(SRC_DIR)/train_random_forest.py
	@V_NUM=$$(ls $(PROCESSED_DIR)/rf-v*.csv 2>/dev/null | wc -l | tr -d ' ' | awk '{print $$1 + 1}'); \
	if [ -f $(PROCESSED_DIR)/submission_rf.csv ]; then \
		mv $(PROCESSED_DIR)/submission_rf.csv $(PROCESSED_DIR)/rf-v$$V_NUM.csv; \
		echo "Success: Created $(PROCESSED_DIR)/rf-v$$V_NUM.csv"; \
	fi

compare:
	@python3 $(SRC_DIR)/compare_results.py

clean:
	rm -rf $(PROCESSED_DIR)/*.csv

vis:
	@echo "Generating Visualizations..."
	python3 src/visualizations.py
	python3 src/visualizations_roc.py
	@echo "✔ Visualizations saved to reports/figures/"

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py 

train:
	python src/models/train.py

eval:
	python -m src.models.evaluate

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload kingabzpro/New-Classification-MLOps ./ / --repo-type=space --commit-message="Sync API files"
deploy: hf-login push-hub

all: install format train eval update-branch deploy
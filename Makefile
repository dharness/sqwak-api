.PHONY: dev

dev:
	flask --app sqwak.__init__:app run --debug -p 8000
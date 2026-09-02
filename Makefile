# llm-stack — common tasks. Run `make` or `make help` to list them.
# Most targets just delegate to ./bin/llm (which wraps the panel's HTTP API).
LLM := ./bin/llm

.PHONY: help start stop restart status logs health models presets url \
        switch engine stop-model install-service firewall kvsweep bench git-init

help:            ## list targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  make %-16s %s\n",$$1,$$2}'

start:           ## start the control panel + router
	@$(LLM) panel start
stop:            ## stop the control panel
	@$(LLM) panel stop
restart:         ## restart the control panel
	@$(LLM) panel restart
status:          ## GPU / running model / engine / health (JSON)
	@$(LLM) status
logs:            ## tail the model server log
	@$(LLM) logs
health:          ## quick router health check
	@$(LLM) health
models:          ## list models discovered under ~/models
	@$(LLM) models
presets:         ## list configured presets
	@$(LLM) presets
url:             ## print the router URL for harnesses
	@$(LLM) url

# usage: make switch PRESET=q8-96k   |   make engine ENGINE=vllm
switch:          ## restart the local model into PRESET=<id>
	@$(LLM) switch $(PRESET)
engine:          ## switch engine ENGINE=llamacpp|vllm|ollama
	@$(LLM) engine $(ENGINE)
stop-model:      ## stop the local model, free the GPU
	@$(LLM) stop-model

install-service: ## install + enable the systemd unit (needs sudo)
	@$(LLM) install-service
firewall:        ## print ufw commands to open the LAN ports
	@$(LLM) firewall

kvsweep:         ## find max context per KV quant on this GPU (benchmark)
	@bash scripts/kvsweep.sh
bench:           ## llama-bench prompt/gen throughput on the current model
	@~/llama.cpp/build/bin/llama-bench -m "$$(find ~/models -iname '*.gguf' ! -iname 'mmproj*' | head -1)" -ngl 99

git-init:        ## first-time: initialise the git repo + initial commit
	@git init -q && git add -A && git commit -qm "Initial commit: local + cloud LLM stack" && echo "committed"

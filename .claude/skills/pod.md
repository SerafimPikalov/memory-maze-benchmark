---
name: pod
description: "Manage RunPod GPU pods (create, status, cost, terminate)"
user_invocable: true
---

Manage RunPod GPU pods for training. This skill delegates to the runpod-manager agent.

If the user says `/pod` without arguments, ask what they want:
- "create" — set up a new GPU pod (will gather requirements first)
- "status" — check running pods: `python runpod/pod_manager.py status`
- "cost" — check spending: `python runpod/pod_manager.py cost`
- "list" — list all pods: `python runpod/pod_manager.py list`
- "gpus" — show available GPUs: `python runpod/pod_manager.py gpus`
- "terminate <id>" — terminate a pod: `python runpod/pod_manager.py terminate <id>`

For "create", use the runpod-manager agent which will ask about:
1. RUNPOD_API_KEY
2. Intent (train, notebooks, smoke test, dev)
3. W&B logging
4. Testing preference
5. Configuration confirmation

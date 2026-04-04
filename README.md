---
title: Email Triage Openenv
emoji: ⚡
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: AI-powered email classification and response environment
---

# Email Triage OpenEnv

## Description
This environment simulates a real-world email classification and response system using OpenEnv.

## Tasks
- Easy: classify spam vs important
- Medium: classify + assign priority
- Hard: generate appropriate response

## Action Space
- classification: spam / important
- priority: low / medium / high
- response: text

## Observation Space
- email_text
- step_count
- history

## Reward Logic
- Correct classification → reward
- Correct priority → reward
- Meaningful response → reward
- Wrong classification → penalty
- Rewards normalized between 0 and 1

## Run Locally

```bash
python inference.py
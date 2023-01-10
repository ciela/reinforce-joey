#!/usr/bin/env bash
python3 -m joeynmt test configs/pg_in_domain.yaml --ckpt pretrained_iwslt14/best.ckpt --alpha 1.0 && \
python3 -m joeynmt test configs/pg_in_domain.yaml --ckpt models/pg_in_domain/pg_in_domain_best.ckpt --alpha 1.0 && \
python3 -m joeynmt test configs/mrt_in_domain.yaml --ckpt models/mrt_in_domain/mrt_in_domain_best.ckpt --alpha 1.0 && \
python3 -m joeynmt test configs/pg_cross_domain.yaml --ckpt pretrained_wmt15/best.ckpt --alpha 1.0 && \
python3 -m joeynmt test configs/pg_cross_domain.yaml --ckpt models/pg_cross_domain/pg_cross_domain_best.ckpt --alpha 1.0 && \
python3 -m joeynmt test configs/mrt_cross_domain.yaml --ckpt models/mrt_cross_domain/mrt_cross_domain_best.ckpt --alpha 1.0

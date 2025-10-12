# Database Schema Snapshot

Generated from the running ParadeDB/Postgres instance (`agentdb`). Tables are grouped by schema and list each column's data type, nullability, and default value if provided.

## Schema: agent

### Table: events

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | bigint | NO | nextval('agent.events_id_seq'::regclass) |
| `run_id` | uuid | YES |  |
| `session_id` | uuid | YES |  |
| `raw` | jsonb | NO |  |
| `created_at` | timestamp with time zone | NO | now() |

### Table: runs

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `user_message` | text | NO |  |
| `user_metadata` | jsonb | YES | '{}'::jsonb |
| `planner_model` | text | YES |  |
| `responder_model` | text | YES |  |
| `audit_model` | text | YES |  |
| `plan` | jsonb | YES |  |
| `response` | jsonb | YES |  |
| `audit_report` | jsonb | YES |  |
| `evidence` | jsonb | YES |  |
| `success` | boolean | YES |  |
| `started_at` | timestamp with time zone | YES | now() |
| `completed_at` | timestamp with time zone | YES |  |
| `updated_at` | timestamp with time zone | YES | now() |
| `duration_ms` | integer | YES |  |
| `chat_ingest_item_id` | uuid | YES |  |
| `metadata` | jsonb | YES | '{}'::jsonb |

### Table: search_sandbox

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO | gen_random_uuid() |
| `run_id` | uuid | NO |  |
| `requirement_id` | text | YES |  |
| `task_id` | text | YES |  |
| `iteration` | integer | NO | 0 |
| `rank` | integer | NO |  |
| `source` | text | YES |  |
| `title` | text | YES |  |
| `url` | text | NO |  |
| `snippet` | text | YES |  |
| `raw_result` | jsonb | NO | '{}'::jsonb |
| `promoted` | boolean | NO | false |
| `promoted_at` | timestamp with time zone | YES |  |
| `kb_document_id` | uuid | YES |  |
| `kb_chunk_ids` | ARRAY | YES |  |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |

### Table: sessions

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `agent_id` | integer | YES |  |
| `meta` | jsonb | NO | '{}'::jsonb |
| `created_at` | timestamp with time zone | NO | now() |

### Table: web_work_items

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `run_id` | uuid | NO |  |
| `requirement_id` | text | YES |  |
| `task_id` | text | YES |  |
| `query` | text | NO |  |
| `source_url` | text | YES |  |
| `source_title` | text | YES |  |
| `raw_result` | jsonb | YES | '{}'::jsonb |
| `snippet` | text | YES |  |
| `query_embedding` | USER-DEFINED | YES |  |
| `snippet_embedding` | USER-DEFINED | YES |  |
| `retrieval_score` | numeric | YES |  |
| `fetch_status` | text | YES |  |
| `http_status` | integer | YES |  |
| `fetched_at` | timestamp with time zone | YES |  |
| `rendered_at` | timestamp with time zone | YES |  |
| `html` | text | YES |  |
| `markdown` | text | YES |  |
| `summary` | text | YES |  |
| `authority_score` | numeric | YES |  |
| `topicality_score` | numeric | YES |  |
| `locality_score` | numeric | YES |  |
| `curated` | boolean | YES | false |
| `curated_reason` | text | YES |  |
| `kb_document_id` | uuid | YES |  |
| `kb_chunk_ids` | ARRAY | YES | ARRAY[]::uuid[] |
| `metadata` | jsonb | YES | '{}'::jsonb |
| `created_at` | timestamp with time zone | YES | now() |
| `updated_at` | timestamp with time zone | YES | now() |
| `expires_at` | timestamp with time zone | YES | (now() + '48:00:00'::interval) |


## Schema: cfg

### Table: active

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `key` | text | NO |  |
| `value` | text | NO |  |
| `updated_at` | timestamp with time zone | NO | now() |

### Table: agent_tools

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `agent_id` | integer | NO |  |
| `tool_id` | integer | NO |  |
| `overrides` | jsonb | NO | '{}'::jsonb |

### Table: agents

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('cfg.agents_id_seq'::regclass) |
| `name` | text | NO |  |
| `model_alias` | text | YES |  |
| `params` | jsonb | NO | '{}'::jsonb |
| `enabled` | boolean | NO | true |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |
| `type` | USER-DEFINED | NO | 'base'::agent_type |
| `system_prompt` | text | YES |  |
| `tools_profile` | text | YES |  |
| `notes` | text | YES |  |
| `db_scope` | jsonb | NO | '[]'::jsonb |

### Table: models

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('cfg.models_id_seq'::regclass) |
| `alias` | text | YES |  |
| `name` | text | NO |  |
| `endpoint` | text | YES |  |
| `dims` | integer | YES |  |
| `purpose` | text | NO | 'chat'::text |
| `enabled` | boolean | NO | true |
| `notes` | text | YES |  |
| `default_params` | jsonb | NO | '{}'::jsonb |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |
| `provider_id` | integer | YES |  |
| `pricing` | jsonb | NO | '{}'::jsonb |
| `provider` | text | YES |  |
| `identifier` | text | YES |  |
| `uri_template` | text | YES |  |
| `version` | text | YES |  |
| `config` | jsonb | YES | '{}'::jsonb |

### Table: policies

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('cfg.policies_id_seq'::regclass) |
| `name` | text | NO |  |
| `description` | text | YES |  |
| `rules` | jsonb | NO | '{}'::jsonb |
| `enabled` | boolean | NO | true |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |

### Table: prompts

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('cfg.prompts_id_seq'::regclass) |
| `name` | text | NO |  |
| `content` | text | NO |  |
| `version` | text | YES |  |
| `metadata` | jsonb | NO | '{}'::jsonb |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |

### Table: providers

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('cfg.providers_id_seq'::regclass) |
| `slug` | text | NO |  |
| `notes` | text | YES |  |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |
| `display_name` | text | YES |  |

### Table: tools

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('cfg.tools_id_seq'::regclass) |
| `slug` | text | NO |  |
| `kind` | USER-DEFINED | NO |  |
| `manifest_or_ref` | text | NO |  |
| `default_params` | jsonb | NO | '{}'::jsonb |
| `enabled` | boolean | NO | true |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |
| `notes` | text | YES |  |


## Schema: ingest

### Table: test

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO |  |
| `full_name` | text | NO |  |
| `email` | text | NO |  |
| `signup_ts` | timestamp with time zone | NO |  |


## Schema: kb

### Table: chunk_embeddings

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `chunk_id` | uuid | NO |  |
| `space_id` | integer | NO |  |
| `embedding` | USER-DEFINED | YES |  |
| `score_meta` | jsonb | YES | '{}'::jsonb |
| `created_at` | timestamp with time zone | YES | now() |
| `version` | integer | NO | 1 |

### Table: chunks

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `ingest_item_id` | uuid | YES |  |
| `document_id` | uuid | YES |  |
| `chunk_index` | integer | NO |  |
| `heading_path` | ARRAY | YES | ARRAY[]::text[] |
| `kind` | text | YES |  |
| `text` | text | NO |  |
| `summary` | text | YES |  |
| `token_count` | integer | YES |  |
| `overlap_tokens` | integer | YES |  |
| `ner_entities` | jsonb | YES | '[]'::jsonb |
| `tsv` | tsvector | YES |  |
| `metadata` | jsonb | YES | '{}'::jsonb |
| `created_at` | timestamp with time zone | YES | now() |
| `version` | integer | NO | 1 |
| `updated_at` | timestamp with time zone | YES | now() |

### Table: document_embeddings

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `document_id` | uuid | NO |  |
| `space_id` | integer | NO |  |
| `embedding` | USER-DEFINED | YES |  |
| `created_at` | timestamp with time zone | YES | now() |
| `version` | integer | NO | 1 |

### Table: documents

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `ingest_item_id` | uuid | YES |  |
| `source_uri` | text | YES |  |
| `file_name` | text | YES |  |
| `title` | text | YES |  |
| `text_full` | text | YES |  |
| `tsv` | tsvector | YES |  |
| `metadata` | jsonb | YES | '{}'::jsonb |
| `summary` | text | YES |  |
| `content_digest` | text | YES |  |
| `version` | integer | NO | 1 |
| `created_at` | timestamp with time zone | YES | now() |
| `updated_at` | timestamp with time zone | YES | now() |

### Table: embedding_spaces

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('kb.embedding_spaces_id_seq'::regclass) |
| `name` | text | NO |  |
| `model` | text | NO |  |
| `provider` | text | NO |  |
| `dims` | integer | NO |  |
| `distance_metric` | text | NO |  |

### Table: entries

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO | gen_random_uuid() |
| `session_id` | uuid | YES |  |
| `ingest_item_id` | uuid | YES |  |
| `document_id` | uuid | YES |  |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |
| `source` | text | NO |  |
| `uri` | text | YES |  |
| `title` | text | YES |  |
| `author` | text | YES |  |
| `content` | text | NO |  |
| `summary` | text | YES |  |
| `meta` | jsonb | NO | '{}'::jsonb |
| `ner` | jsonb | YES |  |
| `emotions` | jsonb | YES |  |
| `embedding` | USER-DEFINED | YES |  |
| `version` | integer | NO | 1 |
| `file_name` | text | YES |  |
| `search_tsv` | tsvector | YES |  |
| `is_document` | boolean | NO | false |
| `is_note` | boolean | NO | false |
| `needs_metadata` | boolean | NO | false |
| `is_chat` | boolean | NO | false |

### Table: entry_metadata

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO | gen_random_uuid() |
| `entry_id` | uuid | NO |  |
| `meta_type` | text | NO |  |
| `data` | jsonb | NO | '{}'::jsonb |
| `created_at` | timestamp with time zone | YES | now() |
| `updated_at` | timestamp with time zone | YES | now() |

### Table: ingest_items

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `job_id` | uuid | YES |  |
| `source_type` | text | NO |  |
| `source_uri` | text | NO |  |
| `display_name` | text | NO |  |
| `mime_type` | text | YES |  |
| `language` | text | YES |  |
| `domain` | text | YES |  |
| `domain_confidence` | numeric | YES |  |
| `status` | text | NO | 'pending'::text |
| `document_summary` | text | YES |  |
| `metadata` | jsonb | YES | '{}'::jsonb |
| `error_info` | jsonb | YES |  |
| `created_at` | timestamp with time zone | YES | now() |
| `updated_at` | timestamp with time zone | YES | now() |


## Schema: paradedb

### Table: index_layer_info

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `relname` | text | YES |  |
| `layer_size` | text | YES |  |
| `low` | numeric | YES |  |
| `high` | numeric | YES |  |
| `byte_size` | numeric | YES |  |
| `count` | bigint | YES |  |
| `segments` | ARRAY | YES |  |


## Schema: public

### Table: DailyTagSpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `individual_request_tag` | text | YES |  |
| `spend_date` | date | YES |  |
| `log_count` | bigint | YES |  |
| `total_spend` | double precision | YES |  |

### Table: Last30dKeysBySpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `api_key` | text | YES |  |
| `key_alias` | text | YES |  |
| `key_name` | text | YES |  |
| `total_spend` | double precision | YES |  |

### Table: Last30dModelsBySpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `model` | text | YES |  |
| `total_spend` | double precision | YES |  |

### Table: Last30dTopEndUsersSpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `end_user` | text | YES |  |
| `total_events` | bigint | YES |  |
| `total_spend` | double precision | YES |  |

### Table: LiteLLM_AuditLog

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `changed_by` | text | NO | ''::text |
| `changed_by_api_key` | text | NO | ''::text |
| `action` | text | NO |  |
| `table_name` | text | NO |  |
| `object_id` | text | NO |  |
| `before_value` | jsonb | YES |  |
| `updated_values` | jsonb | YES |  |

### Table: LiteLLM_BudgetTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `budget_id` | text | NO |  |
| `max_budget` | double precision | YES |  |
| `soft_budget` | double precision | YES |  |
| `max_parallel_requests` | integer | YES |  |
| `tpm_limit` | bigint | YES |  |
| `rpm_limit` | bigint | YES |  |
| `model_max_budget` | jsonb | YES |  |
| `budget_duration` | text | YES |  |
| `budget_reset_at` | timestamp without time zone | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | NO |  |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_by` | text | NO |  |

### Table: LiteLLM_Config

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `param_name` | text | NO |  |
| `param_value` | jsonb | YES |  |

### Table: LiteLLM_CredentialsTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `credential_id` | text | NO |  |
| `credential_name` | text | NO |  |
| `credential_values` | jsonb | NO |  |
| `credential_info` | jsonb | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | NO |  |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_by` | text | NO |  |

### Table: LiteLLM_CronJob

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `cronjob_id` | text | NO |  |
| `pod_id` | text | NO |  |
| `status` | USER-DEFINED | NO | 'INACTIVE'::"JobStatus" |
| `last_updated` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `ttl` | timestamp without time zone | NO |  |

### Table: LiteLLM_DailyTagSpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `tag` | text | YES |  |
| `date` | text | NO |  |
| `api_key` | text | NO |  |
| `model` | text | YES |  |
| `model_group` | text | YES |  |
| `custom_llm_provider` | text | YES |  |
| `mcp_namespaced_tool_name` | text | YES |  |
| `prompt_tokens` | bigint | NO | 0 |
| `completion_tokens` | bigint | NO | 0 |
| `cache_read_input_tokens` | bigint | NO | 0 |
| `cache_creation_input_tokens` | bigint | NO | 0 |
| `spend` | double precision | NO | 0.0 |
| `api_requests` | bigint | NO | 0 |
| `successful_requests` | bigint | NO | 0 |
| `failed_requests` | bigint | NO | 0 |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |

### Table: LiteLLM_DailyTeamSpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `team_id` | text | YES |  |
| `date` | text | NO |  |
| `api_key` | text | NO |  |
| `model` | text | YES |  |
| `model_group` | text | YES |  |
| `custom_llm_provider` | text | YES |  |
| `mcp_namespaced_tool_name` | text | YES |  |
| `prompt_tokens` | bigint | NO | 0 |
| `completion_tokens` | bigint | NO | 0 |
| `cache_read_input_tokens` | bigint | NO | 0 |
| `cache_creation_input_tokens` | bigint | NO | 0 |
| `spend` | double precision | NO | 0.0 |
| `api_requests` | bigint | NO | 0 |
| `successful_requests` | bigint | NO | 0 |
| `failed_requests` | bigint | NO | 0 |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |

### Table: LiteLLM_DailyUserSpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | YES |  |
| `date` | text | NO |  |
| `api_key` | text | NO |  |
| `model` | text | YES |  |
| `model_group` | text | YES |  |
| `custom_llm_provider` | text | YES |  |
| `mcp_namespaced_tool_name` | text | YES |  |
| `prompt_tokens` | bigint | NO | 0 |
| `completion_tokens` | bigint | NO | 0 |
| `cache_read_input_tokens` | bigint | NO | 0 |
| `cache_creation_input_tokens` | bigint | NO | 0 |
| `spend` | double precision | NO | 0.0 |
| `api_requests` | bigint | NO | 0 |
| `successful_requests` | bigint | NO | 0 |
| `failed_requests` | bigint | NO | 0 |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |

### Table: LiteLLM_EndUserTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `user_id` | text | NO |  |
| `alias` | text | YES |  |
| `spend` | double precision | NO | 0.0 |
| `allowed_model_region` | text | YES |  |
| `default_model` | text | YES |  |
| `budget_id` | text | YES |  |
| `blocked` | boolean | NO | false |

### Table: LiteLLM_ErrorLogs

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `request_id` | text | NO |  |
| `startTime` | timestamp without time zone | NO |  |
| `endTime` | timestamp without time zone | NO |  |
| `api_base` | text | NO | ''::text |
| `model_group` | text | NO | ''::text |
| `litellm_model_name` | text | NO | ''::text |
| `model_id` | text | NO | ''::text |
| `request_kwargs` | jsonb | NO | '{}'::jsonb |
| `exception_type` | text | NO | ''::text |
| `exception_string` | text | NO | ''::text |
| `status_code` | text | NO | ''::text |

### Table: LiteLLM_GuardrailsTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `guardrail_id` | text | NO |  |
| `guardrail_name` | text | NO |  |
| `litellm_params` | jsonb | NO |  |
| `guardrail_info` | jsonb | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |

### Table: LiteLLM_HealthCheckTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `health_check_id` | text | NO |  |
| `model_name` | text | NO |  |
| `model_id` | text | YES |  |
| `status` | text | NO |  |
| `healthy_count` | integer | NO | 0 |
| `unhealthy_count` | integer | NO | 0 |
| `error_message` | text | YES |  |
| `response_time_ms` | double precision | YES |  |
| `details` | jsonb | YES |  |
| `checked_by` | text | YES |  |
| `checked_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |

### Table: LiteLLM_InvitationLink

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `is_accepted` | boolean | NO | false |
| `accepted_at` | timestamp without time zone | YES |  |
| `expires_at` | timestamp without time zone | NO |  |
| `created_at` | timestamp without time zone | NO |  |
| `created_by` | text | NO |  |
| `updated_at` | timestamp without time zone | NO |  |
| `updated_by` | text | NO |  |

### Table: LiteLLM_MCPServerTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `server_id` | text | NO |  |
| `server_name` | text | YES |  |
| `alias` | text | YES |  |
| `description` | text | YES |  |
| `url` | text | YES |  |
| `transport` | text | NO | 'sse'::text |
| `auth_type` | text | YES |  |
| `created_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |
| `created_by` | text | YES |  |
| `updated_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |
| `updated_by` | text | YES |  |
| `mcp_info` | jsonb | YES | '{}'::jsonb |
| `mcp_access_groups` | ARRAY | YES |  |
| `status` | text | YES | 'unknown'::text |
| `last_health_check` | timestamp without time zone | YES |  |
| `health_check_error` | text | YES |  |
| `command` | text | YES |  |
| `args` | ARRAY | YES | ARRAY[]::text[] |
| `env` | jsonb | YES | '{}'::jsonb |

### Table: LiteLLM_ManagedFileTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `unified_file_id` | text | NO |  |
| `file_object` | jsonb | YES |  |
| `model_mappings` | jsonb | NO |  |
| `flat_model_file_ids` | ARRAY | YES | ARRAY[]::text[] |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | YES |  |
| `updated_at` | timestamp without time zone | NO |  |
| `updated_by` | text | YES |  |

### Table: LiteLLM_ManagedObjectTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `unified_object_id` | text | NO |  |
| `model_object_id` | text | NO |  |
| `file_object` | jsonb | NO |  |
| `file_purpose` | text | NO |  |
| `status` | text | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | YES |  |
| `updated_at` | timestamp without time zone | NO |  |
| `updated_by` | text | YES |  |

### Table: LiteLLM_ManagedVectorStoresTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `vector_store_id` | text | NO |  |
| `custom_llm_provider` | text | NO |  |
| `vector_store_name` | text | YES |  |
| `vector_store_description` | text | YES |  |
| `vector_store_metadata` | jsonb | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |
| `litellm_credential_name` | text | YES |  |
| `litellm_params` | jsonb | YES |  |

### Table: LiteLLM_ModelTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('"LiteLLM_ModelTable_id_seq"'::regclass) |
| `aliases` | jsonb | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | NO |  |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_by` | text | NO |  |

### Table: LiteLLM_ObjectPermissionTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `object_permission_id` | text | NO |  |
| `mcp_servers` | ARRAY | YES | ARRAY[]::text[] |
| `mcp_access_groups` | ARRAY | YES | ARRAY[]::text[] |
| `vector_stores` | ARRAY | YES | ARRAY[]::text[] |

### Table: LiteLLM_OrganizationMembership

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `user_id` | text | NO |  |
| `organization_id` | text | NO |  |
| `user_role` | text | YES |  |
| `spend` | double precision | YES | 0.0 |
| `budget_id` | text | YES |  |
| `created_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |

### Table: LiteLLM_OrganizationTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `organization_id` | text | NO |  |
| `organization_alias` | text | NO |  |
| `budget_id` | text | NO |  |
| `metadata` | jsonb | NO | '{}'::jsonb |
| `models` | ARRAY | YES |  |
| `spend` | double precision | NO | 0.0 |
| `model_spend` | jsonb | NO | '{}'::jsonb |
| `object_permission_id` | text | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | NO |  |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_by` | text | NO |  |

### Table: LiteLLM_PromptTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `prompt_id` | text | NO |  |
| `litellm_params` | jsonb | NO |  |
| `prompt_info` | jsonb | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO |  |

### Table: LiteLLM_ProxyModelTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `model_id` | text | NO |  |
| `model_name` | text | NO |  |
| `litellm_params` | jsonb | NO |  |
| `model_info` | jsonb | YES |  |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `created_by` | text | NO |  |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_by` | text | NO |  |

### Table: LiteLLM_SpendLogs

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `request_id` | text | NO |  |
| `call_type` | text | NO |  |
| `api_key` | text | NO | ''::text |
| `spend` | double precision | NO | 0.0 |
| `total_tokens` | integer | NO | 0 |
| `prompt_tokens` | integer | NO | 0 |
| `completion_tokens` | integer | NO | 0 |
| `startTime` | timestamp without time zone | NO |  |
| `endTime` | timestamp without time zone | NO |  |
| `completionStartTime` | timestamp without time zone | YES |  |
| `model` | text | NO | ''::text |
| `model_id` | text | YES | ''::text |
| `model_group` | text | YES | ''::text |
| `custom_llm_provider` | text | YES | ''::text |
| `api_base` | text | YES | ''::text |
| `user` | text | YES | ''::text |
| `metadata` | jsonb | YES | '{}'::jsonb |
| `cache_hit` | text | YES | ''::text |
| `cache_key` | text | YES | ''::text |
| `request_tags` | jsonb | YES | '[]'::jsonb |
| `team_id` | text | YES |  |
| `end_user` | text | YES |  |
| `requester_ip_address` | text | YES |  |
| `messages` | jsonb | YES | '{}'::jsonb |
| `response` | jsonb | YES | '{}'::jsonb |
| `session_id` | text | YES |  |
| `status` | text | YES |  |
| `mcp_namespaced_tool_name` | text | YES |  |
| `proxy_server_request` | jsonb | YES | '{}'::jsonb |

### Table: LiteLLM_TeamMembership

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `user_id` | text | NO |  |
| `team_id` | text | NO |  |
| `spend` | double precision | NO | 0.0 |
| `budget_id` | text | YES |  |

### Table: LiteLLM_TeamTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `team_id` | text | NO |  |
| `team_alias` | text | YES |  |
| `organization_id` | text | YES |  |
| `object_permission_id` | text | YES |  |
| `admins` | ARRAY | YES |  |
| `members` | ARRAY | YES |  |
| `members_with_roles` | jsonb | NO | '{}'::jsonb |
| `metadata` | jsonb | NO | '{}'::jsonb |
| `max_budget` | double precision | YES |  |
| `spend` | double precision | NO | 0.0 |
| `models` | ARRAY | YES |  |
| `max_parallel_requests` | integer | YES |  |
| `tpm_limit` | bigint | YES |  |
| `rpm_limit` | bigint | YES |  |
| `budget_duration` | text | YES |  |
| `budget_reset_at` | timestamp without time zone | YES |  |
| `blocked` | boolean | NO | false |
| `created_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | NO | CURRENT_TIMESTAMP |
| `model_spend` | jsonb | NO | '{}'::jsonb |
| `model_max_budget` | jsonb | NO | '{}'::jsonb |
| `team_member_permissions` | ARRAY | YES | ARRAY[]::text[] |
| `model_id` | integer | YES |  |

### Table: LiteLLM_UserNotifications

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `request_id` | text | NO |  |
| `user_id` | text | NO |  |
| `models` | ARRAY | YES |  |
| `justification` | text | NO |  |
| `status` | text | NO |  |

### Table: LiteLLM_UserTable

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `user_id` | text | NO |  |
| `user_alias` | text | YES |  |
| `team_id` | text | YES |  |
| `sso_user_id` | text | YES |  |
| `organization_id` | text | YES |  |
| `object_permission_id` | text | YES |  |
| `password` | text | YES |  |
| `teams` | ARRAY | YES | ARRAY[]::text[] |
| `user_role` | text | YES |  |
| `max_budget` | double precision | YES |  |
| `spend` | double precision | NO | 0.0 |
| `user_email` | text | YES |  |
| `models` | ARRAY | YES |  |
| `metadata` | jsonb | NO | '{}'::jsonb |
| `max_parallel_requests` | integer | YES |  |
| `tpm_limit` | bigint | YES |  |
| `rpm_limit` | bigint | YES |  |
| `budget_duration` | text | YES |  |
| `budget_reset_at` | timestamp without time zone | YES |  |
| `allowed_cache_controls` | ARRAY | YES | ARRAY[]::text[] |
| `model_spend` | jsonb | NO | '{}'::jsonb |
| `model_max_budget` | jsonb | NO | '{}'::jsonb |
| `created_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |
| `updated_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |

### Table: LiteLLM_VerificationToken

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `token` | text | NO |  |
| `key_name` | text | YES |  |
| `key_alias` | text | YES |  |
| `soft_budget_cooldown` | boolean | NO | false |
| `spend` | double precision | NO | 0.0 |
| `expires` | timestamp without time zone | YES |  |
| `models` | ARRAY | YES |  |
| `aliases` | jsonb | NO | '{}'::jsonb |
| `config` | jsonb | NO | '{}'::jsonb |
| `user_id` | text | YES |  |
| `team_id` | text | YES |  |
| `permissions` | jsonb | NO | '{}'::jsonb |
| `max_parallel_requests` | integer | YES |  |
| `metadata` | jsonb | NO | '{}'::jsonb |
| `blocked` | boolean | YES |  |
| `tpm_limit` | bigint | YES |  |
| `rpm_limit` | bigint | YES |  |
| `max_budget` | double precision | YES |  |
| `budget_duration` | text | YES |  |
| `budget_reset_at` | timestamp without time zone | YES |  |
| `allowed_cache_controls` | ARRAY | YES | ARRAY[]::text[] |
| `allowed_routes` | ARRAY | YES | ARRAY[]::text[] |
| `model_spend` | jsonb | NO | '{}'::jsonb |
| `model_max_budget` | jsonb | NO | '{}'::jsonb |
| `budget_id` | text | YES |  |
| `organization_id` | text | YES |  |
| `object_permission_id` | text | YES |  |
| `created_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |
| `created_by` | text | YES |  |
| `updated_at` | timestamp without time zone | YES | CURRENT_TIMESTAMP |
| `updated_by` | text | YES |  |

### Table: LiteLLM_VerificationTokenView

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `token` | text | YES |  |
| `key_name` | text | YES |  |
| `key_alias` | text | YES |  |
| `soft_budget_cooldown` | boolean | YES |  |
| `spend` | double precision | YES |  |
| `expires` | timestamp without time zone | YES |  |
| `models` | ARRAY | YES |  |
| `aliases` | jsonb | YES |  |
| `config` | jsonb | YES |  |
| `user_id` | text | YES |  |
| `team_id` | text | YES |  |
| `permissions` | jsonb | YES |  |
| `max_parallel_requests` | integer | YES |  |
| `metadata` | jsonb | YES |  |
| `blocked` | boolean | YES |  |
| `tpm_limit` | bigint | YES |  |
| `rpm_limit` | bigint | YES |  |
| `max_budget` | double precision | YES |  |
| `budget_duration` | text | YES |  |
| `budget_reset_at` | timestamp without time zone | YES |  |
| `allowed_cache_controls` | ARRAY | YES |  |
| `allowed_routes` | ARRAY | YES |  |
| `model_spend` | jsonb | YES |  |
| `model_max_budget` | jsonb | YES |  |
| `budget_id` | text | YES |  |
| `organization_id` | text | YES |  |
| `object_permission_id` | text | YES |  |
| `created_at` | timestamp without time zone | YES |  |
| `created_by` | text | YES |  |
| `updated_at` | timestamp without time zone | YES |  |
| `updated_by` | text | YES |  |
| `team_spend` | double precision | YES |  |
| `team_max_budget` | double precision | YES |  |
| `team_tpm_limit` | bigint | YES |  |
| `team_rpm_limit` | bigint | YES |  |

### Table: MonthlyGlobalSpend

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `date` | date | YES |  |
| `spend` | double precision | YES |  |

### Table: MonthlyGlobalSpendPerKey

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `date` | date | YES |  |
| `spend` | double precision | YES |  |
| `api_key` | text | YES |  |

### Table: MonthlyGlobalSpendPerUserPerKey

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `date` | date | YES |  |
| `spend` | double precision | YES |  |
| `api_key` | text | YES |  |
| `user` | text | YES |  |

### Table: _prisma_migrations

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `checksum` | character varying | NO |  |
| `finished_at` | timestamp with time zone | YES |  |
| `migration_name` | character varying | NO |  |
| `logs` | text | YES |  |
| `rolled_back_at` | timestamp with time zone | YES |  |
| `started_at` | timestamp with time zone | NO | now() |
| `applied_steps_count` | integer | NO | 0 |

### Table: alembic_version

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `version_num` | character varying | NO |  |

### Table: auth

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `email` | character varying | NO |  |
| `password` | text | NO |  |
| `active` | boolean | NO |  |

### Table: channel

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | YES |  |
| `name` | text | YES |  |
| `description` | text | YES |  |
| `data` | json | YES |  |
| `meta` | json | YES |  |
| `access_control` | json | YES |  |
| `created_at` | bigint | YES |  |
| `updated_at` | bigint | YES |  |
| `type` | text | YES |  |

### Table: channel_member

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `channel_id` | text | NO |  |
| `user_id` | text | NO |  |
| `created_at` | bigint | YES |  |

### Table: chat

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `user_id` | character varying | NO |  |
| `title` | text | NO |  |
| `share_id` | character varying | YES |  |
| `archived` | boolean | NO |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |
| `chat` | json | YES |  |
| `pinned` | boolean | YES |  |
| `meta` | json | NO | '{}'::json |
| `folder_id` | text | YES |  |

### Table: chatidtag

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `tag_name` | character varying | NO |  |
| `chat_id` | character varying | NO |  |
| `user_id` | character varying | NO |  |
| `timestamp` | bigint | NO |  |

### Table: chunk_tags

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `chunk_id` | uuid | YES |  |
| `tag_key` | text | YES |  |
| `tag_value` | text | YES |  |

### Table: chunks

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `doc_id` | uuid | YES |  |
| `doc_version` | integer | NO | 1 |
| `version` | integer | NO | 1 |
| `section` | text | YES |  |
| `idx` | integer | YES |  |
| `text` | text | YES |  |
| `page_from` | integer | YES |  |
| `page_to` | integer | YES |  |
| `token_count` | integer | YES |  |
| `overlap_used` | integer | YES |  |
| `emb_general` | USER-DEFINED | YES |  |
| `emb_legal` | USER-DEFINED | YES |  |
| `created_at` | timestamp without time zone | YES | now() |

### Table: config

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('config_id_seq'::regclass) |
| `data` | json | NO |  |
| `version` | integer | NO |  |
| `created_at` | timestamp without time zone | NO | now() |
| `updated_at` | timestamp without time zone | YES | now() |

### Table: doc_tags

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `doc_id` | uuid | YES |  |
| `tag_key` | text | YES |  |
| `tag_value` | text | YES |  |

### Table: docs

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `slug` | text | NO |  |
| `version` | integer | NO | 1 |
| `source_file` | text | YES |  |
| `kind` | text | YES |  |
| `content_md` | text | YES |  |
| `token_count` | integer | YES |  |
| `emb_general` | USER-DEFINED | YES |  |
| `emb_legal` | USER-DEFINED | YES |  |
| `created_at` | timestamp without time zone | YES | now() |

### Table: document

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('document_id_seq'::regclass) |
| `collection_name` | character varying | NO |  |
| `name` | character varying | NO |  |
| `title` | text | NO |  |
| `filename` | text | NO |  |
| `content` | text | YES |  |
| `user_id` | character varying | NO |  |
| `timestamp` | bigint | NO |  |

### Table: feedback

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | YES |  |
| `version` | bigint | YES |  |
| `type` | text | YES |  |
| `data` | json | YES |  |
| `meta` | json | YES |  |
| `snapshot` | json | YES |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |

### Table: file

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `filename` | text | NO |  |
| `meta` | json | YES |  |
| `created_at` | bigint | NO |  |
| `hash` | text | YES |  |
| `data` | json | YES |  |
| `updated_at` | bigint | YES |  |
| `path` | text | YES |  |
| `access_control` | json | YES |  |

### Table: folder

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `parent_id` | text | YES |  |
| `user_id` | text | NO |  |
| `name` | text | NO |  |
| `items` | json | YES |  |
| `meta` | json | YES |  |
| `is_expanded` | boolean | NO |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |
| `data` | json | YES |  |

### Table: function

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `name` | text | NO |  |
| `type` | text | NO |  |
| `content` | text | NO |  |
| `meta` | text | NO |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |
| `valves` | text | YES |  |
| `is_active` | boolean | NO |  |
| `is_global` | boolean | NO |  |

### Table: geography_columns

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `f_table_catalog` | name | YES |  |
| `f_table_schema` | name | YES |  |
| `f_table_name` | name | YES |  |
| `f_geography_column` | name | YES |  |
| `coord_dimension` | integer | YES |  |
| `srid` | integer | YES |  |
| `type` | text | YES |  |

### Table: geometry_columns

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `f_table_catalog` | character varying | YES |  |
| `f_table_schema` | name | YES |  |
| `f_table_name` | name | YES |  |
| `f_geometry_column` | name | YES |  |
| `coord_dimension` | integer | YES |  |
| `srid` | integer | YES |  |
| `type` | character varying | YES |  |

### Table: group

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | YES |  |
| `name` | text | YES |  |
| `description` | text | YES |  |
| `data` | json | YES |  |
| `meta` | json | YES |  |
| `permissions` | json | YES |  |
| `user_ids` | json | YES |  |
| `created_at` | bigint | YES |  |
| `updated_at` | bigint | YES |  |

### Table: knowledge

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `name` | text | NO |  |
| `description` | text | YES |  |
| `data` | json | YES |  |
| `meta` | json | YES |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | YES |  |
| `access_control` | json | YES |  |

### Table: memory

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `user_id` | character varying | NO |  |
| `content` | text | NO |  |
| `updated_at` | bigint | NO |  |
| `created_at` | bigint | NO |  |

### Table: message

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | YES |  |
| `channel_id` | text | YES |  |
| `content` | text | YES |  |
| `data` | json | YES |  |
| `meta` | json | YES |  |
| `created_at` | bigint | YES |  |
| `updated_at` | bigint | YES |  |
| `parent_id` | text | YES |  |

### Table: message_reaction

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `message_id` | text | NO |  |
| `name` | text | NO |  |
| `created_at` | bigint | YES |  |

### Table: migratehistory

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('migratehistory_id_seq'::regclass) |
| `name` | character varying | NO |  |
| `migrated_at` | timestamp without time zone | NO |  |

### Table: model

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `base_model_id` | text | YES |  |
| `name` | text | NO |  |
| `meta` | text | NO |  |
| `params` | text | NO |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |
| `access_control` | json | YES |  |
| `is_active` | boolean | NO | true |

### Table: note

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | YES |  |
| `title` | text | YES |  |
| `data` | json | YES |  |
| `meta` | json | YES |  |
| `access_control` | json | YES |  |
| `created_at` | bigint | YES |  |
| `updated_at` | bigint | YES |  |

### Table: oauth_session

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `provider` | text | NO |  |
| `token` | text | NO |  |
| `expires_at` | bigint | NO |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |

### Table: prompt

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('prompt_id_seq'::regclass) |
| `command` | character varying | NO |  |
| `user_id` | character varying | NO |  |
| `title` | text | NO |  |
| `content` | text | NO |  |
| `timestamp` | bigint | NO |  |
| `access_control` | json | YES |  |

### Table: spatial_ref_sys

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `srid` | integer | NO |  |
| `auth_name` | character varying | YES |  |
| `auth_srid` | integer | YES |  |
| `srtext` | character varying | YES |  |
| `proj4text` | character varying | YES |  |

### Table: tag

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `name` | character varying | NO |  |
| `user_id` | character varying | NO |  |
| `meta` | json | YES |  |

### Table: test

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `counter` | integer | NO | nextval('test_counter_seq'::regclass) |
| `test` | text | NO |  |

### Table: tool

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | text | NO |  |
| `user_id` | text | NO |  |
| `name` | text | NO |  |
| `content` | text | NO |  |
| `specs` | text | NO |  |
| `meta` | text | NO |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |
| `valves` | text | YES |  |
| `access_control` | json | YES |  |

### Table: user

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | character varying | NO |  |
| `name` | character varying | NO |  |
| `email` | character varying | NO |  |
| `role` | character varying | NO |  |
| `profile_image_url` | text | NO |  |
| `api_key` | character varying | YES |  |
| `created_at` | bigint | NO |  |
| `updated_at` | bigint | NO |  |
| `last_active_at` | bigint | NO |  |
| `settings` | text | YES |  |
| `info` | text | YES |  |
| `oauth_sub` | text | YES |  |
| `username` | character varying | YES |  |
| `bio` | text | YES |  |
| `gender` | text | YES |  |
| `date_of_birth` | date | YES |  |

### Table: web

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `url` | text | NO |  |
| `slug` | text | NO |  |
| `version` | integer | NO | 1 |
| `title` | text | YES |  |
| `content_md` | text | YES |  |
| `token_count` | integer | YES |  |
| `emb_general` | USER-DEFINED | YES |  |
| `emb_legal` | USER-DEFINED | YES |  |
| `retrieved_at` | timestamp without time zone | YES | now() |

### Table: web_chunk_tags

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `chunk_id` | uuid | YES |  |
| `tag_key` | text | YES |  |
| `tag_value` | text | YES |  |

### Table: web_chunks

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | uuid | NO |  |
| `web_id` | uuid | YES |  |
| `web_version` | integer | NO | 1 |
| `version` | integer | NO | 1 |
| `section` | text | YES |  |
| `idx` | integer | YES |  |
| `text` | text | YES |  |
| `token_count` | integer | YES |  |
| `overlap_used` | integer | YES |  |
| `emb_general` | USER-DEFINED | YES |  |
| `emb_legal` | USER-DEFINED | YES |  |
| `created_at` | timestamp without time zone | YES | now() |

### Table: web_tags

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `web_id` | uuid | YES |  |
| `tag_key` | text | YES |  |
| `tag_value` | text | YES |  |


## Schema: tiger

### Table: addr

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.addr_gid_seq'::regclass) |
| `tlid` | bigint | YES |  |
| `fromhn` | character varying | YES |  |
| `tohn` | character varying | YES |  |
| `side` | character varying | YES |  |
| `zip` | character varying | YES |  |
| `plus4` | character varying | YES |  |
| `fromtyp` | character varying | YES |  |
| `totyp` | character varying | YES |  |
| `fromarmid` | integer | YES |  |
| `toarmid` | integer | YES |  |
| `arid` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `statefp` | character varying | YES |  |

### Table: addrfeat

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.addrfeat_gid_seq'::regclass) |
| `tlid` | bigint | YES |  |
| `statefp` | character varying | NO |  |
| `aridl` | character varying | YES |  |
| `aridr` | character varying | YES |  |
| `linearid` | character varying | YES |  |
| `fullname` | character varying | YES |  |
| `lfromhn` | character varying | YES |  |
| `ltohn` | character varying | YES |  |
| `rfromhn` | character varying | YES |  |
| `rtohn` | character varying | YES |  |
| `zipl` | character varying | YES |  |
| `zipr` | character varying | YES |  |
| `edge_mtfcc` | character varying | YES |  |
| `parityl` | character varying | YES |  |
| `parityr` | character varying | YES |  |
| `plus4l` | character varying | YES |  |
| `plus4r` | character varying | YES |  |
| `lfromtyp` | character varying | YES |  |
| `ltotyp` | character varying | YES |  |
| `rfromtyp` | character varying | YES |  |
| `rtotyp` | character varying | YES |  |
| `offsetl` | character varying | YES |  |
| `offsetr` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: bg

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.bg_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `tractce` | character varying | YES |  |
| `blkgrpce` | character varying | YES |  |
| `bg_id` | character varying | NO |  |
| `namelsad` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | double precision | YES |  |
| `awater` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: county

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.county_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `countyns` | character varying | YES |  |
| `cntyidfp` | character varying | NO |  |
| `name` | character varying | YES |  |
| `namelsad` | character varying | YES |  |
| `lsad` | character varying | YES |  |
| `classfp` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `csafp` | character varying | YES |  |
| `cbsafp` | character varying | YES |  |
| `metdivfp` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | bigint | YES |  |
| `awater` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: county_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `st_code` | integer | NO |  |
| `state` | character varying | YES |  |
| `co_code` | integer | NO |  |
| `name` | character varying | YES |  |

### Table: countysub_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `st_code` | integer | NO |  |
| `state` | character varying | YES |  |
| `co_code` | integer | NO |  |
| `county` | character varying | YES |  |
| `cs_code` | integer | NO |  |
| `name` | character varying | YES |  |

### Table: cousub

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.cousub_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `cousubfp` | character varying | YES |  |
| `cousubns` | character varying | YES |  |
| `cosbidfp` | character varying | NO |  |
| `name` | character varying | YES |  |
| `namelsad` | character varying | YES |  |
| `lsad` | character varying | YES |  |
| `classfp` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `cnectafp` | character varying | YES |  |
| `nectafp` | character varying | YES |  |
| `nctadvfp` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | numeric | YES |  |
| `awater` | numeric | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: direction_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `name` | character varying | NO |  |
| `abbrev` | character varying | YES |  |

### Table: edges

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.edges_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `tlid` | bigint | YES |  |
| `tfidl` | numeric | YES |  |
| `tfidr` | numeric | YES |  |
| `mtfcc` | character varying | YES |  |
| `fullname` | character varying | YES |  |
| `smid` | character varying | YES |  |
| `lfromadd` | character varying | YES |  |
| `ltoadd` | character varying | YES |  |
| `rfromadd` | character varying | YES |  |
| `rtoadd` | character varying | YES |  |
| `zipl` | character varying | YES |  |
| `zipr` | character varying | YES |  |
| `featcat` | character varying | YES |  |
| `hydroflg` | character varying | YES |  |
| `railflg` | character varying | YES |  |
| `roadflg` | character varying | YES |  |
| `olfflg` | character varying | YES |  |
| `passflg` | character varying | YES |  |
| `divroad` | character varying | YES |  |
| `exttyp` | character varying | YES |  |
| `ttyp` | character varying | YES |  |
| `deckedroad` | character varying | YES |  |
| `artpath` | character varying | YES |  |
| `persist` | character varying | YES |  |
| `gcseflg` | character varying | YES |  |
| `offsetl` | character varying | YES |  |
| `offsetr` | character varying | YES |  |
| `tnidf` | numeric | YES |  |
| `tnidt` | numeric | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: faces

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.faces_gid_seq'::regclass) |
| `tfid` | numeric | YES |  |
| `statefp00` | character varying | YES |  |
| `countyfp00` | character varying | YES |  |
| `tractce00` | character varying | YES |  |
| `blkgrpce00` | character varying | YES |  |
| `blockce00` | character varying | YES |  |
| `cousubfp00` | character varying | YES |  |
| `submcdfp00` | character varying | YES |  |
| `conctyfp00` | character varying | YES |  |
| `placefp00` | character varying | YES |  |
| `aiannhfp00` | character varying | YES |  |
| `aiannhce00` | character varying | YES |  |
| `comptyp00` | character varying | YES |  |
| `trsubfp00` | character varying | YES |  |
| `trsubce00` | character varying | YES |  |
| `anrcfp00` | character varying | YES |  |
| `elsdlea00` | character varying | YES |  |
| `scsdlea00` | character varying | YES |  |
| `unsdlea00` | character varying | YES |  |
| `uace00` | character varying | YES |  |
| `cd108fp` | character varying | YES |  |
| `sldust00` | character varying | YES |  |
| `sldlst00` | character varying | YES |  |
| `vtdst00` | character varying | YES |  |
| `zcta5ce00` | character varying | YES |  |
| `tazce00` | character varying | YES |  |
| `ugace00` | character varying | YES |  |
| `puma5ce00` | character varying | YES |  |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `tractce` | character varying | YES |  |
| `blkgrpce` | character varying | YES |  |
| `blockce` | character varying | YES |  |
| `cousubfp` | character varying | YES |  |
| `submcdfp` | character varying | YES |  |
| `conctyfp` | character varying | YES |  |
| `placefp` | character varying | YES |  |
| `aiannhfp` | character varying | YES |  |
| `aiannhce` | character varying | YES |  |
| `comptyp` | character varying | YES |  |
| `trsubfp` | character varying | YES |  |
| `trsubce` | character varying | YES |  |
| `anrcfp` | character varying | YES |  |
| `ttractce` | character varying | YES |  |
| `tblkgpce` | character varying | YES |  |
| `elsdlea` | character varying | YES |  |
| `scsdlea` | character varying | YES |  |
| `unsdlea` | character varying | YES |  |
| `uace` | character varying | YES |  |
| `cd111fp` | character varying | YES |  |
| `sldust` | character varying | YES |  |
| `sldlst` | character varying | YES |  |
| `vtdst` | character varying | YES |  |
| `zcta5ce` | character varying | YES |  |
| `tazce` | character varying | YES |  |
| `ugace` | character varying | YES |  |
| `puma5ce` | character varying | YES |  |
| `csafp` | character varying | YES |  |
| `cbsafp` | character varying | YES |  |
| `metdivfp` | character varying | YES |  |
| `cnectafp` | character varying | YES |  |
| `nectafp` | character varying | YES |  |
| `nctadvfp` | character varying | YES |  |
| `lwflag` | character varying | YES |  |
| `offset` | character varying | YES |  |
| `atotal` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |
| `tractce20` | character varying | YES |  |
| `blkgrpce20` | character varying | YES |  |
| `blockce20` | character varying | YES |  |
| `countyfp20` | character varying | YES |  |
| `statefp20` | character varying | YES |  |

### Table: featnames

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.featnames_gid_seq'::regclass) |
| `tlid` | bigint | YES |  |
| `fullname` | character varying | YES |  |
| `name` | character varying | YES |  |
| `predirabrv` | character varying | YES |  |
| `pretypabrv` | character varying | YES |  |
| `prequalabr` | character varying | YES |  |
| `sufdirabrv` | character varying | YES |  |
| `suftypabrv` | character varying | YES |  |
| `sufqualabr` | character varying | YES |  |
| `predir` | character varying | YES |  |
| `pretyp` | character varying | YES |  |
| `prequal` | character varying | YES |  |
| `sufdir` | character varying | YES |  |
| `suftyp` | character varying | YES |  |
| `sufqual` | character varying | YES |  |
| `linearid` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `paflag` | character varying | YES |  |
| `statefp` | character varying | YES |  |

### Table: geocode_settings

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `name` | text | NO |  |
| `setting` | text | YES |  |
| `unit` | text | YES |  |
| `category` | text | YES |  |
| `short_desc` | text | YES |  |

### Table: geocode_settings_default

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `name` | text | NO |  |
| `setting` | text | YES |  |
| `unit` | text | YES |  |
| `category` | text | YES |  |
| `short_desc` | text | YES |  |

### Table: loader_lookuptables

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `process_order` | integer | NO | 1000 |
| `lookup_name` | text | NO |  |
| `table_name` | text | YES |  |
| `single_mode` | boolean | NO | true |
| `load` | boolean | NO | true |
| `level_county` | boolean | NO | false |
| `level_state` | boolean | NO | false |
| `level_nation` | boolean | NO | false |
| `post_load_process` | text | YES |  |
| `single_geom_mode` | boolean | YES | false |
| `insert_mode` | character | NO | 'c'::bpchar |
| `pre_load_process` | text | YES |  |
| `columns_exclude` | ARRAY | YES |  |
| `website_root_override` | text | YES |  |

### Table: loader_platform

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `os` | character varying | NO |  |
| `declare_sect` | text | YES |  |
| `pgbin` | text | YES |  |
| `wget` | text | YES |  |
| `unzip_command` | text | YES |  |
| `psql` | text | YES |  |
| `path_sep` | text | YES |  |
| `loader` | text | YES |  |
| `environ_set_command` | text | YES |  |
| `county_process_command` | text | YES |  |

### Table: loader_variables

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `tiger_year` | character varying | NO |  |
| `website_root` | text | YES |  |
| `staging_fold` | text | YES |  |
| `data_schema` | text | YES |  |
| `staging_schema` | text | YES |  |

### Table: pagc_gaz

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('tiger.pagc_gaz_id_seq'::regclass) |
| `seq` | integer | YES |  |
| `word` | text | YES |  |
| `stdword` | text | YES |  |
| `token` | integer | YES |  |
| `is_custom` | boolean | NO | true |

### Table: pagc_lex

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('tiger.pagc_lex_id_seq'::regclass) |
| `seq` | integer | YES |  |
| `word` | text | YES |  |
| `stdword` | text | YES |  |
| `token` | integer | YES |  |
| `is_custom` | boolean | NO | true |

### Table: pagc_rules

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('tiger.pagc_rules_id_seq'::regclass) |
| `rule` | text | YES |  |
| `is_custom` | boolean | YES | true |

### Table: place

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.place_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `placefp` | character varying | YES |  |
| `placens` | character varying | YES |  |
| `plcidfp` | character varying | NO |  |
| `name` | character varying | YES |  |
| `namelsad` | character varying | YES |  |
| `lsad` | character varying | YES |  |
| `classfp` | character varying | YES |  |
| `cpi` | character varying | YES |  |
| `pcicbsa` | character varying | YES |  |
| `pcinecta` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | bigint | YES |  |
| `awater` | bigint | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: place_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `st_code` | integer | NO |  |
| `state` | character varying | YES |  |
| `pl_code` | integer | NO |  |
| `name` | character varying | YES |  |

### Table: secondary_unit_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `name` | character varying | NO |  |
| `abbrev` | character varying | YES |  |

### Table: state

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.state_gid_seq'::regclass) |
| `region` | character varying | YES |  |
| `division` | character varying | YES |  |
| `statefp` | character varying | NO |  |
| `statens` | character varying | YES |  |
| `stusps` | character varying | NO |  |
| `name` | character varying | YES |  |
| `lsad` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | bigint | YES |  |
| `awater` | bigint | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: state_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `st_code` | integer | NO |  |
| `name` | character varying | YES |  |
| `abbrev` | character varying | YES |  |
| `statefp` | character | YES |  |

### Table: street_type_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `name` | character varying | NO |  |
| `abbrev` | character varying | YES |  |
| `is_hw` | boolean | NO | false |

### Table: tabblock

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.tabblock_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `tractce` | character varying | YES |  |
| `blockce` | character varying | YES |  |
| `tabblock_id` | character varying | NO |  |
| `name` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `ur` | character varying | YES |  |
| `uace` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | double precision | YES |  |
| `awater` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: tabblock20

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `tractce` | character varying | YES |  |
| `blockce` | character varying | YES |  |
| `geoid` | character varying | NO |  |
| `name` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `ur` | character varying | YES |  |
| `uace` | character varying | YES |  |
| `uatype` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | double precision | YES |  |
| `awater` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |
| `housing` | double precision | YES |  |
| `pop` | double precision | YES |  |

### Table: tract

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.tract_gid_seq'::regclass) |
| `statefp` | character varying | YES |  |
| `countyfp` | character varying | YES |  |
| `tractce` | character varying | YES |  |
| `tract_id` | character varying | NO |  |
| `name` | character varying | YES |  |
| `namelsad` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | double precision | YES |  |
| `awater` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: zcta5

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `gid` | integer | NO | nextval('tiger.zcta5_gid_seq'::regclass) |
| `statefp` | character varying | NO |  |
| `zcta5ce` | character varying | NO |  |
| `classfp` | character varying | YES |  |
| `mtfcc` | character varying | YES |  |
| `funcstat` | character varying | YES |  |
| `aland` | double precision | YES |  |
| `awater` | double precision | YES |  |
| `intptlat` | character varying | YES |  |
| `intptlon` | character varying | YES |  |
| `partflg` | character varying | YES |  |
| `the_geom` | USER-DEFINED | YES |  |

### Table: zip_lookup

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `zip` | integer | NO |  |
| `st_code` | integer | YES |  |
| `state` | character varying | YES |  |
| `co_code` | integer | YES |  |
| `county` | character varying | YES |  |
| `cs_code` | integer | YES |  |
| `cousub` | character varying | YES |  |
| `pl_code` | integer | YES |  |
| `place` | character varying | YES |  |
| `cnt` | integer | YES |  |

### Table: zip_lookup_all

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `zip` | integer | YES |  |
| `st_code` | integer | YES |  |
| `state` | character varying | YES |  |
| `co_code` | integer | YES |  |
| `county` | character varying | YES |  |
| `cs_code` | integer | YES |  |
| `cousub` | character varying | YES |  |
| `pl_code` | integer | YES |  |
| `place` | character varying | YES |  |
| `cnt` | integer | YES |  |

### Table: zip_lookup_base

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `zip` | character varying | NO |  |
| `state` | character varying | YES |  |
| `county` | character varying | YES |  |
| `city` | character varying | YES |  |
| `statefp` | character varying | YES |  |

### Table: zip_state

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `zip` | character varying | NO |  |
| `stusps` | character varying | NO |  |
| `statefp` | character varying | YES |  |

### Table: zip_state_loc

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `zip` | character varying | NO |  |
| `stusps` | character varying | NO |  |
| `statefp` | character varying | YES |  |
| `place` | character varying | NO |  |


## Schema: topology

### Table: layer

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `topology_id` | integer | NO |  |
| `layer_id` | integer | NO |  |
| `schema_name` | character varying | NO |  |
| `table_name` | character varying | NO |  |
| `feature_column` | character varying | NO |  |
| `feature_type` | integer | NO |  |
| `level` | integer | NO | 0 |
| `child_id` | integer | YES |  |

### Table: topology

| Column | Data type | Nullable | Default |
|--------|-----------|----------|---------|
| `id` | integer | NO | nextval('topology.topology_id_seq'::regclass) |
| `name` | character varying | NO |  |
| `srid` | integer | NO |  |
| `precision` | double precision | NO |  |
| `hasz` | boolean | NO | false |
| `useslargeids` | boolean | NO | false |


## Schema: (134 rows)

### Table: None

_No column data available._


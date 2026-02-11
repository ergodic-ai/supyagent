---
name: supy
description: >-
  Use supyagent to interact with cloud services. Connected: Google (list emails from gmail inbox, get a specific email by its message id, send an email via gmail, list upcoming events from google calendar); AI Inbox (list events from the ai inbox, get a single inbox event by id, archive a single inbox event by id, batch update inbox events); Resend (send a transactional email via resend, get a sent email by id from resend); Slack (list slack channels in the workspace, get messages from a slack channel, send a message to a slack channel or user, list users in the slack workspace). Use when the user asks to send messages, emails, or interact with connected services.
---

# Supyagent Cloud Integrations

Execute tools: `supyagent service run <tool_name> '<json>'`

Output: `{"ok": true, "data": ...}` on success, `{"ok": false, "error": "..."}` on failure.

---

## Google

### gmail_list_messages

List emails from Gmail inbox. Supports search queries using Gmail syntax.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `maxResults` | integer | no | Number of messages to return (default: 10) |
| `q` | string | no | Search query using Gmail syntax (e.g. 'from:boss@company.com') |
| `pageToken` | string | no | Token for pagination |

```bash
supyagent service run gmail_list_messages '{"maxResults": 10}'
```

### gmail_get_message

Get a specific email by its message ID. Returns full message with body, headers, and metadata.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messageId` | string | yes | The Gmail message ID |

```bash
supyagent service run gmail_get_message '{"messageId": "abc123"}'
```

### gmail_send_message

Send an email via Gmail.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | yes | Recipient email address |
| `subject` | string | yes | Email subject line |
| `body` | string | yes | Email body content (plain text or HTML) |
| `cc` | string | no | CC recipient (optional) |
| `bcc` | string | no | BCC recipient (optional) |
| `isHtml` | boolean | no | Set to true to send body as HTML (default: false) |

```bash
supyagent service run gmail_send_message '{"to": "user@example.com", "subject": "Example", "body": "Hello world", "cc": "user@example.com"}'
```

### calendar_list_events

List upcoming events from Google Calendar.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeMin` | string | no | Start time in ISO format (default: now) |
| `timeMax` | string | no | End time in ISO format |
| `maxResults` | integer | no | Max events to return (default: 10) |
| `calendarId` | string | no | Calendar ID (default: 'primary') |

```bash
supyagent service run calendar_list_events '{"timeMin": "..."}'
```

### calendar_get_event

Get details of a specific calendar event by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eventId` | string | yes | The calendar event ID |
| `calendarId` | string | no | Calendar ID (default: 'primary') |

```bash
supyagent service run calendar_get_event '{"eventId": "abc123", "calendarId": "abc123"}'
```

### calendar_check_freebusy

Check free/busy status for a time range.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeMin` | string | yes | Start time in ISO format |
| `timeMax` | string | yes | End time in ISO format |
| `items` | array | no | Calendar IDs to check |

```bash
supyagent service run calendar_check_freebusy '{"timeMin": "...", "timeMax": "...", "items": []}'
```

### calendar_create_event

Create a new event on Google Calendar.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `summary` | string | yes | Event title |
| `description` | string | no | Event description |
| `start` | object | yes | Start time object with dateTime and timeZone |
| `end` | object | yes | End time object with dateTime and timeZone |
| `attendees` | array | no | List of attendee email addresses |
| `location` | string | no | Event location |

```bash
supyagent service run calendar_create_event '{"summary": "Example", "description": "...", "start": {}, "end": {}}'
```

### calendar_update_event

Update an existing calendar event.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eventId` | string | yes | The calendar event ID |
| `summary` | string | no | Updated event title |
| `description` | string | no | Updated event description |
| `start` | object | no | Updated start time |
| `end` | object | no | Updated end time |
| `location` | string | no | Updated location |

```bash
supyagent service run calendar_update_event '{"eventId": "abc123", "summary": "Example"}'
```

### calendar_delete_event

Delete a calendar event.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eventId` | string | yes | The calendar event ID |
| `calendarId` | string | no | Calendar ID (default: 'primary') |

```bash
supyagent service run calendar_delete_event '{"eventId": "abc123", "calendarId": "abc123"}'
```

### drive_list_files

List files from Google Drive.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pageSize` | integer | no | Number of files to return (default: 20) |
| `folderId` | string | no | List files in a specific folder |
| `orderBy` | string | no | Sort order (default: 'modifiedTime desc') |

```bash
supyagent service run drive_list_files '{"pageSize": 10}'
```

### drive_get_file_content

Download the content of a file from Google Drive.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `fileId` | string | yes | The Drive file ID |
| `format` | string | no | 'base64' (default) or 'raw' |

```bash
supyagent service run drive_get_file_content '{"fileId": "abc123", "format": "base64"}'
```

### drive_search_files

Search for files in Google Drive.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Search query |
| `pageSize` | integer | no | Number of results (default: 10) |
| `mimeType` | string | no | Filter by MIME type (e.g. 'application/pdf') |

```bash
supyagent service run drive_search_files '{"query": "search term", "pageSize": 10}'
```

### drive_upload_file

Upload a file or create a folder in Google Drive.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | yes | File or folder name |
| `mimeType` | string | yes | MIME type (use 'application/vnd.google-apps.folder' for folders) |
| `content` | string | no | Base64-encoded file content (not needed for folders) |

```bash
supyagent service run drive_upload_file '{"name": "Example", "mimeType": "...", "content": "Hello world"}'
```

### drive_delete_file

Delete a file from Google Drive.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `fileId` | string | yes | The Drive file ID |

```bash
supyagent service run drive_delete_file '{"fileId": "abc123"}'
```

---

## AI Inbox

### inbox_list

List events from the AI Inbox. Returns webhook events from connected integrations (GitHub, Slack, Twilio, etc). Supports filtering by status, provider, and event type.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | no | Filter by status: unread, read, archived (default: unread) |
| `provider` | string | no | Filter by provider (e.g. github, slack, twilio) |
| `event_type` | string | no | Filter by event type (e.g. pull_request.opened, message.received) |
| `limit` | integer | no | Number of events to return (default: 20, max: 100) |
| `offset` | integer | no | Offset for pagination (default: 0) |

```bash
supyagent service run inbox_list '{"status": "unread"}'
```

### inbox_get

Get a single inbox event by ID. Returns the full payload. Automatically marks the event as read.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | yes | The event ID |

```bash
supyagent service run inbox_get '{"id": "abc123"}'
```

### inbox_archive

Archive a single inbox event by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | yes | The event ID to archive |

```bash
supyagent service run inbox_archive '{"id": "abc123"}'
```

### inbox_batch

Batch update inbox events. Mark multiple events as read or archived at once.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ids` | array | yes | Array of event IDs |
| `action` | string | yes | Action to perform: read or archive |

```bash
supyagent service run inbox_batch '{"ids": [], "action": "read"}'
```

### inbox_archive_all

Archive all inbox events. Optionally filter by provider or date.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | no | Only archive events from this provider |
| `before` | string | no | Only archive events before this ISO date |

```bash
supyagent service run inbox_archive_all '{"provider": "abc123"}'
```

---

## Resend

### resend_send_email

Send a transactional email via Resend.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | yes | Recipient email address |
| `subject` | string | yes | Email subject line |
| `html` | string | yes | Email body (HTML) |
| `from` | string | no | Sender email (optional, uses account default) |
| `cc` | string | no | CC recipients (comma-separated) |
| `bcc` | string | no | BCC recipients (comma-separated) |
| `replyTo` | string | no | Reply-to email address |
| `text` | string | no | Plain text version of the email |

```bash
supyagent service run resend_send_email '{"to": "user@example.com", "subject": "Example", "html": "...", "from": "..."}'
```

### resend_get_email

Get a sent email by ID from Resend.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `emailId` | string | yes | The Resend email ID |

```bash
supyagent service run resend_get_email '{"emailId": "user@example.com"}'
```

---

## Slack

### slack_list_channels

List Slack channels in the workspace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | no | Number of channels (default: 100) |
| `cursor` | string | no | Pagination cursor |
| `exclude_archived` | boolean | no | Exclude archived channels (default: true) |

```bash
supyagent service run slack_list_channels '{"limit": 10}'
```

### slack_get_channel_messages

Get messages from a Slack channel.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `channelId` | string | yes | Slack channel ID |
| `limit` | integer | no | Number of messages (default: 20) |
| `cursor` | string | no | Pagination cursor |
| `oldest` | string | no | Start of time range (Unix timestamp) |
| `latest` | string | no | End of time range (Unix timestamp) |

```bash
supyagent service run slack_get_channel_messages '{"channelId": "C0123456789", "limit": 10}'
```

### slack_send_message

Send a message to a Slack channel or user.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `channel` | string | yes | Channel ID (C...) or user ID (U...) for DMs |
| `text` | string | yes | Message text |
| `thread_ts` | string | no | Thread timestamp to reply to (optional) |

```bash
supyagent service run slack_send_message '{"channel": "C0123456789", "text": "Hello world", "thread_ts": "..."}'
```

### slack_list_users

List users in the Slack workspace.

```bash
supyagent service run slack_list_users '{}'
```

### slack_get_user

Get a Slack user by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `userId` | string | yes | Slack user ID |

```bash
supyagent service run slack_get_user '{"userId": "abc123"}'
```

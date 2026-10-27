#!/usr/bin/env node
/*
 * agent.js — a pluggable agent harness where every subsystem is a tool.
 *
 * The Node twin of agent.py: same nine components (skills, tools, memory,
 * context, hooks, control/guardrails, orchestration loop, prompts,
 * tasks/scheduler), same tool names, same action contract, same funnel.
 * One file, no dependencies, no build step.
 *
 *     const { Agent } = require('./agent.js');
 *
 *     const agent = new Agent({ root: '.' });        // scripted model by default
 *     agent.script([
 *       { thought: 'look', call: { tool: 'fs.read', args: { path: 'notes.md' } } },
 *       { thought: 'answered', final: 'two notes' },
 *     ]);
 *     const run = await agent.run('what is in the notes?');
 *     console.log(run.status, run.trail);
 *
 * Every turn the model answers with ONE JSON object:
 *
 *     {"thought": "...", "call": {"tool": "<name>", "args": {...}}}   // act
 *     {"thought": "...", "final": "..."}                              // stop
 *
 * Plugging in your own:
 *
 *     const { tool } = require('./agent.js');
 *     module.exports = tool('csv.rows', 'read a csv into rows',
 *                           { args: { path: 'string' }, required: ['path'] },
 *                           ({ path }) => ({ rows: fs.readFileSync(path, 'utf8').split('\n') }));
 *
 *     discover(['./tools'])   // plain files, and packages exporting agentTools/agentModels
 *
 * Nothing executes outside one funnel: registry -> pre_tool hooks -> policy ->
 * handler -> post_tool hooks -> context. Guardrails are deny-first name patterns,
 * a path sandbox around `root`, an approval list, per-run write/delete/call/time
 * budgets, network off by default, and a dry-run that refuses every mutating tool.
 *
 * cli:  node agent.js demo | tools | skills | panel | examples [name] | run "<goal>" [--model model.ollama]
 */
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const DEFAULT_TOOLS_DIR = 'tools';
const ENTRY_POINT_GROUP = 'agentTools';
const MODEL_GROUP = 'agentModels';
const PATH_KEYS = ['path', 'file', 'dir', 'from', 'to', 'cwd', 'root'];
const JSON_TYPES = { string: 'string', integer: 'integer', number: 'number', boolean: 'boolean', array: 'array', object: 'object' };
const TASK_STATUSES = ['pending', 'in_progress', 'completed', 'blocked', 'deleted'];
const HOOK_EVENTS = ['plan', 'pre_tool', 'post_tool', 'tool_error', 'stop'];
const ANY = '*';
const NL = '\n';

const USAGE = [
  'agent.js - a pluggable agent harness where every subsystem is a tool',
  '',
  '  node agent.js demo                       guarded loop over a scratch workspace',
  '  node agent.js tools                      list the registry with danger/mutation/network marks',
  '  node agent.js skills                     list skill packs found on disk',
  '  node agent.js panel                      dump the live control panel as JSON',
  '  node agent.js examples [name ...]        run the example pack (all by default)',
  '  node agent.js run "<goal>" [--model m]   one loop against model.scripted (offline) or an endpoint',
].join(NL);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const estimateTokens = (text) => Math.max(1, Math.floor(String(text ?? '').length / 4));

const resolvePath = (candidate, root) => {
  const raw = String(candidate).replace(/^~(?=$|\/|\\)/, os.homedir());
  return path.isAbsolute(raw) || /^[A-Za-z]:[\\/]/.test(raw) ? raw : path.join(String(root), raw);
};

const under = (target, base) => {
  const child = path.resolve(String(target));
  const parent = path.resolve(String(base)).replace(/[\\/]+$/, '');
  if (!parent) return true;                      // the filesystem root contains everything
  return child === path.resolve(parent) || child.startsWith(path.resolve(parent) + path.sep);
};

function coerce(value, type) {
  /* Tool args arrive as JSON from a model; fit them to the declared type. */
  if (value === null || value === undefined || !type || type === 'any') return value ?? null;
  switch (type) {
    case 'string':
      return typeof value === 'string' ? value : JSON.stringify(value);
    case 'boolean':
      if (typeof value === 'boolean') return value;
      return ['1', 'true', 'yes', 'y', 'on'].includes(String(value).trim().toLowerCase());
    case 'integer': {
      const num = Number(typeof value === 'string' ? value.trim() : value);
      if (!Number.isFinite(num)) throw new Error(`cannot use ${JSON.stringify(value)} as an integer`);
      return Math.trunc(num);
    }
    case 'number': {
      const num = Number(typeof value === 'string' ? value.trim() : value);
      if (!Number.isFinite(num)) throw new Error(`cannot use ${JSON.stringify(value)} as a number`);
      return num;
    }
    case 'array': {
      if (typeof value === 'string') {
        let parsed;
        try { parsed = JSON.parse(value); } catch { return value.split(',').map((part) => part.trim()).filter(Boolean); }
        return Array.isArray(parsed) ? parsed : [parsed];
      }
      if (Array.isArray(value)) return value;
      if (typeof value === 'object') return Object.values(value);
      return [value];
    }
    case 'object': {
      if (typeof value !== 'string') {
        if (value && typeof value === 'object' && !Array.isArray(value)) return value;
        throw new Error(`cannot use ${typeof value} as an object`);
      }
      const parsed = value.trim() ? JSON.parse(value) : {};
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('expected a JSON object');
      return parsed;
    }
    default:
      return value;
  }
}

function firstJson(text) {
  /* Pull the first balanced {...} object out of a model reply. */
  const raw = String(text ?? '');
  let start = raw.indexOf('{');
  while (start !== -1) {
    let depth = 0, inString = false, escape = false;
    for (let index = start; index < raw.length; index += 1) {
      const char = raw[index];
      if (escape) { escape = false; continue; }
      if (inString) {
        if (char === '\\') escape = true;
        else if (char === '"') inString = false;
      } else if (char === '"') inString = true;
      else if (char === '{') depth += 1;
      else if (char === '}') {
        depth -= 1;
        if (depth === 0) {
          let parsed;
          try { parsed = JSON.parse(raw.slice(start, index + 1)); } catch { break; }
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
        }
      }
    }
    start = raw.indexOf('{', start + 1);
  }
  return null;
}

const TAG_RE = /<[^>]+>/g;
const NOISE_RE = /<(script|style)[^>]*>[\s\S]*?<\/\1>/gi;
const TITLE_RE = /<title[^>]*>([\s\S]*?)<\/title>/i;
const LINK_RE = /<a[^>]*href=["']?([^"'> ]+)["']?[^>]*>([\s\S]*?)<\/a>/gi;
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', nbsp: ' ', '#39': "'", '#x27': "'", '#039': "'" };
const unescapeText = (value) => String(value).replace(/&(amp|lt|gt|quot|nbsp|#39|#x27|#039);?/gi,
  (whole, name) => ENTITIES[String(name).toLowerCase()] ?? whole);

function htmlReduce(markup) {
  /* A page, cut down to the three things an agent reads: title, running text, links. */
  const stripped = String(markup ?? '').replace(NOISE_RE, ' ');
  const flat = (value) => unescapeText(value.replace(TAG_RE, ' ')).replace(/\s+/g, ' ').trim();
  const found = TITLE_RE.exec(stripped);
  const body = found ? stripped.replace(found[0], ' ') : stripped;
  const links = [];
  for (const match of body.matchAll(LINK_RE)) {
    if (links.length >= 20) break;
    links.push({ href: match[1], text: flat(match[2]) });
  }
  return { title: found ? flat(found[1]) : '', text: flat(body).slice(0, 8000), links };
}

function globToRegExp(pattern) {
  /* fnmatch semantics: * any run, ? one char, [seq] a class. */
  let out = '';
  for (const char of String(pattern)) {
    if (char === '*') out += '[\\s\\S]*';
    else if (char === '?') out += '[\\s\\S]';
    else if ('.+^${}()|\\'.includes(char)) out += `\\${char}`;
    else out += char;
  }
  try { return new RegExp(`^${out}$`, 'u'); }
  catch { return new RegExp(`^${String(pattern).split('').map((c) => ('.+^${}()|[]\\'.includes(c) ? `\\${c}` : c)).join('')}$`); }
}

const fnmatch = (name, pattern) => globToRegExp(pattern).test(String(name));

const matchesAny = (name, patterns) => (patterns || []).some((pattern) => fnmatch(name, pattern));

const nowSeconds = () => Date.now() / 1000;

const clone = (value) => JSON.parse(JSON.stringify(value ?? null));

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

class ToolError extends Error {}          // a tool ran and failed; the message joins the transcript
class Blocked extends Error {}            // a hook or guardrail refused the call
class UnknownTool extends Error {}
class SandboxError extends Error {}       // path or budget violation detected by Policy

// ---------------------------------------------------------------------------
// tools — the one interface everything speaks
// ---------------------------------------------------------------------------

class Tool {
  constructor({ name, desc = '', handler, args = {}, required = [], kind = 'tool',
                mutating = false, network = false, danger = false, deletes = false, plugin = 'core' }) {
    if (!name) throw new ToolError('a tool needs a name');
    Object.assign(this, { name, desc: desc || name, handler, args, kind, mutating, network, danger, deletes, plugin });
    this.required = Array.isArray(required) ? required : [...(required || [])];
  }

  static argType(hint) {
    if (Array.isArray(hint)) return typeof hint[0] === 'string' ? hint[0] : 'string';
    return typeof hint === 'string' ? hint : 'string';
  }

  schema() {
    const props = {};
    for (const [key, hint] of Object.entries(this.args)) {
      const description = typeof hint === 'string' ? hint : (Array.isArray(hint) && hint[1]) || '';
      props[key] = { type: JSON_TYPES[Tool.argType(hint)] || 'string', description: description || this.desc };
    }
    return { type: 'object', properties: props, required: [...this.required] };
  }

  openaiSpec() {
    return { type: 'function', function: { name: this.name, description: this.desc, parameters: this.schema() } };
  }

  async call(args) {
    const bound = {};
    for (const [key, value] of Object.entries(args || {})) {
      if (!(key in this.args)) {
        throw new ToolError(`${this.name} got unexpected arg '${key}'; it takes `
          + `${Object.keys(this.args).length ? JSON.stringify(Object.keys(this.args).sort()) : 'nothing'}`);
      }
      try { bound[key] = coerce(value, Tool.argType(this.args[key])); }
      catch (error) { throw new ToolError(`${this.name} arg '${key}': ${error.message}`); }
    }
    const missing = this.required.filter((key) => !(key in bound));
    if (missing.length) throw new ToolError(`${this.name} is missing required arg(s): ${missing.join(', ')}`);
    return this.handler(bound);
  }

  brief() {
    const args = {};
    for (const [key, hint] of Object.entries(this.args)) {
      args[key] = typeof hint === 'string' ? hint : (Array.isArray(hint) ? hint[0] : 'string');
    }
    return { name: this.name, kind: this.kind, plugin: this.plugin, desc: this.desc, args,
             required: [...this.required], mutating: this.mutating, network: this.network,
             danger: this.danger, deletes: this.deletes };
  }
}

class Registry {
  /* Flat name -> Tool. Components register here; the loop dispatches only from here. */
  constructor(tools = {}) { this.tools = { ...tools }; }

  get length() { return Object.keys(this.tools).length; }

  add(spec) { this.tools[spec.name] = spec; return spec; }

  remove(name) { const had = name in this.tools; delete this.tools[name]; return had; }

  get(name) {
    if (!(name in this.tools)) {
      throw new UnknownTool(`no tool '${name}'. ${this.length} are registered; tools.list shows them`);
    }
    return this.tools[name];
  }

  has(name) { return name in this.tools; }

  names(kind = null) {
    return Object.values(this.tools).filter((t) => !kind || t.kind === kind).map((t) => t.name).sort();
  }

  brief(kind = null) { return this.names(kind).map((name) => this.tools[name].brief()); }
}

const defaultRegistry = new Registry();

function tool(...rest) {
  /* tool(name, desc, opts, fn) or tool(name, opts, fn) — registers a pluggable tool. */
  const fn = rest.pop();
  const [first, second, third] = rest;
  const name = first || (fn && fn.name) || 'tool';
  let desc = '', opts = {};
  if (typeof second === 'string') { desc = second; opts = third || {}; }
  else if (second && typeof second === 'object') { opts = second; }
  if (typeof first === 'object' && first !== null) { opts = first; desc = opts.desc || ''; }
  const spec = new Tool({
    name: typeof first === 'string' ? name : (opts.name || name),
    desc: desc || opts.desc || String(fn.doc || '').trim().split('\n')[0] || name,
    handler: fn,
    args: opts.args || {},
    required: opts.required || [],
    kind: opts.kind || 'tool',
    mutating: !!opts.mutating,
    network: !!opts.network,
    danger: !!opts.danger,
    deletes: !!opts.deletes,
    plugin: opts.plugin || 'core',
  });
  (opts.registry || defaultRegistry).add(spec);
  return spec;
}

// A plugin that requires this file while agent.js is still the module under evaluation
// would otherwise be handed an empty export object, so publish its surface here.
Object.assign(module.exports, { Tool, Registry, defaultRegistry, tool,
                                ToolError, Blocked, UnknownTool, SandboxError });

function adopt(target, candidate, fallback, loaded) {
  /* A plugin may hand us a Tool, or a function carrying .toolName/.toolArgs metadata. */
  if (candidate instanceof Tool) {
    if (!target.has(candidate.name)) target.add(candidate);
    loaded.push(candidate.name);
    return;
  }
  if (typeof candidate !== 'function') return;
  const meta = candidate.tool || {};
  const name = meta.name || candidate.toolName || fallback;
  const args = meta.args || candidate.toolArgs || {};
  target.add(new Tool({
    name, desc: meta.desc || candidate.doc || candidate.toolDesc || name, handler: candidate, args,
    required: meta.required || candidate.toolRequired || Object.keys(args),
    kind: meta.kind || 'tool', mutating: !!(meta.mutating || candidate.toolMutating),
    network: !!(meta.network || candidate.toolNetwork), danger: !!meta.danger, deletes: !!meta.deletes,
    plugin: meta.plugin || fallback,
  }));
  loaded.push(name);
}

const flatten = (exported) => {
  if (exported instanceof Tool || typeof exported === 'function') return [exported];
  if (Array.isArray(exported)) return exported.flatMap(flatten);
  if (exported && typeof exported === 'object') return Object.entries(exported).map(([key, item]) => [item, key]);
  return [];
};

const pluginModules = {};   // module stem -> tool names it registered, remembered across discovers

function discover(paths = null, options = {}) {
  /* Import plugin files from ./tools, plus packages exporting agentTools/agentModels. */
  const target = options.registry || defaultRegistry;
  const roots = paths === null || paths === undefined ? [DEFAULT_TOOLS_DIR]
    : (Array.isArray(paths) ? paths : [paths]);
  const loaded = [], modules = [], errors = [];
  const added = {};

  const handOver = (names) => {
    for (const name of names) {
      if (defaultRegistry.has(name) && !target.has(name)) {
        target.add(defaultRegistry.tools[name]);
        added[name] = true;
      }
    }
  };

  for (const raw of roots) {
    const root = path.resolve(String(raw).replace(/^~(?=$|\/|\\)/, os.homedir()));
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) continue;
    for (const file of fs.readdirSync(root).filter((name) => name.endsWith('.js') && !name.startsWith('_')).sort()) {
      const stem = path.basename(file, '.js');
      const before = Object.keys(defaultRegistry.tools);
      const rows = [];
      try {
        const exported = require(path.join(root, file));
        for (const row of flatten(exported && exported.default ? exported.default : exported)) {
          if (Array.isArray(row)) adopt(target, row[0], row[1] || stem, rows);
          else adopt(target, row, stem, rows);
        }
      } catch (error) {
        errors.push([stem, `${error.name}: ${error.message}`]);
        continue;
      }
      pluginModules[stem] = [...new Set([...(pluginModules[stem] || []),
                                         ...Object.keys(defaultRegistry.tools).filter((n) => !before.includes(n)),
                                         ...rows])];
      handOver(pluginModules[stem]);
      loaded.push(...rows.filter((name) => target.has(name)));
      modules.push(stem);
    }
  }
  for (const group of [ENTRY_POINT_GROUP, MODEL_GROUP]) {
    for (const pack of options.entryPoints ? options.entryPoints(group) : []) {
      const rows = [];
      try {
        for (const row of flatten(pack.value)) {
          if (Array.isArray(row)) adopt(target, row[0], row[1] || pack.name, rows);
          else adopt(target, row, pack.name, rows);
        }
      } catch (error) { errors.push([pack.name, `${error.name}: ${error.message}`]); continue; }
      handOver(rows);
      loaded.push(...rows.filter((name) => target.has(name)));
    }
  }
  return { loaded: [...new Set([...Object.keys(added), ...loaded])], modules,
           errors: Object.fromEntries(errors) };
}

// ---------------------------------------------------------------------------
// control and guard rails — deny wins over allow, always
// ---------------------------------------------------------------------------

// The operator's own handles. A rule list must never be able to lock the person who
// typed it out of the guardrails, so calls from these sources skip the gate for these
// tools. Everything the model drives is enforced in full.
const GUARDRAIL_TOOLS = ['policy.', 'confirm.'];
const OPERATOR_SOURCES = ['panel', 'cli', 'operator'];

class Policy {
  constructor(options = {}) {
    Object.assign(this, {
      root: path.resolve(String(options.root || process.cwd())),
      allow: [],                                     // empty means "everything not denied"
      deny: ['agent.run', 'fs.delete', 'shell.*'],
      confirm_required: ['fs.write', 'fs.edit', 'shell.run'],
      forbidden_commands: ['rm -rf /', 'rm -rf ~', 'mkfs', 'dd if=', ':(){', 'sudo', 'shutdown', 'reboot',
        'reg delete', '> /dev/', '|sh', '| powershell', 'del /f /s /q', 'format ', 'del c:\\', 'chmod -R 777'],
      extra_dirs: [], max_steps: 12, max_tool_calls: 60, max_writes: 20, max_deletes: 2, max_seconds: 300,
      allow_network: false, dry_run: false, granted: {}, calls: 0, writes: 0, deletes: 0, started_at: 0, blocks: [],
    });
    for (const [key, value] of Object.entries(options)) if (Object.hasOwn(this, key)) this[key] = value;
    this.extra_dirs = (options.extra_dirs || []).map(String);
  }

  startRun() { this.calls = this.writes = this.deletes = 0; this.started_at = nowSeconds(); }  // grants survive

  grant(name, repeats = false) { this.granted[name] = repeats ? -1 : 0; }

  revoke(name) { const had = name in this.granted; delete this.granted[name]; return had; }

  block(name, reason) { this.blocks.push({ tool: name, reason, at: nowSeconds() }); return [false, reason]; }

  check(spec, args = {}, source = 'loop') {
    const name = spec.name;
    if (OPERATOR_SOURCES.includes(source) && GUARDRAIL_TOOLS.some((prefix) => name.startsWith(prefix))) {
      return [true, ''];
    }
    for (const pattern of this.deny) {
      if (fnmatch(name, pattern)) return this.block(name, `denied by guardrail rule '${pattern}'`);
    }
    if (this.allow.length && !matchesAny(name, this.allow)) {
      return this.block(name, `not in the allow list ${JSON.stringify(this.allow)}`);
    }
    if (spec.network && !this.allow_network) {
      return this.block(name, 'network tools are off (set policy.allow_network=True)');
    }
    const command = String((args || {}).cmd === undefined ? '' : args.cmd);
    if (command) {
      const hit = this.forbidden_commands.find((pattern) => command.includes(pattern));
      if (hit) return this.block(name, `command pattern '${hit}' is forbidden`);
    }
    if (this.dry_run && spec.mutating) return this.block(name, 'dry-run refuses mutating tools');
    if (this.confirm_required.includes(name)) {
      const left = this.granted[name];
      if (left === undefined) return this.block(name, `${name} needs approval: call confirm.grant(tool='${name}')`);
      if (left === 0) delete this.granted[name];
      else this.granted[name] = left - 1;
    }
    if (spec.mutating && this.calls >= this.max_tool_calls) {
      return this.block(name, `tool-call budget ${this.max_tool_calls} already spent`);
    }
    try { this.charge(spec, args || {}); } catch (error) {
      if (error instanceof SandboxError) return this.block(name, error.message);
      throw error;
    }
    if (this.started_at && this.max_seconds && nowSeconds() - this.started_at > this.max_seconds) {
      return this.block(name, `wall-clock limit ${this.max_seconds}s reached`);
    }
    this.calls += 1;
    return [true, ''];
  }

  charge(spec, args) {
    const bases = [this.root, ...this.extra_dirs.map(String)];
    for (const key of PATH_KEYS) {
      if (args[key] !== undefined && args[key] !== null && args[key] !== '') {
        const target = path.resolve(resolvePath(args[key], this.root));
        if (!bases.some((base) => under(target, base))) {
          throw new SandboxError(`arg ${key}='${args[key]}' escapes the sandbox ${this.root}`);
        }
      }
    }
    if (!spec.mutating) return;
    if (spec.deletes) {
      if (this.deletes >= this.max_deletes) throw new SandboxError(`delete budget ${this.max_deletes} already spent`);
      this.deletes += 1;
    } else {
      if (this.writes >= this.max_writes) throw new SandboxError(`write budget ${this.max_writes} already spent`);
      this.writes += 1;
    }
  }

  set(changes) {
    for (const [key, value] of Object.entries(changes || {})) {
      if (!Object.hasOwn(this, key)) throw new ToolError(`no such policy knob '${key}'`);
      this[key] = key === 'root' ? path.resolve(String(value)) : value;
    }
    return this.summary();
  }

  summary() {
    return { root: this.root, allow: this.allow, deny: this.deny, confirm_required: this.confirm_required,
             granted: this.granted, allow_network: this.allow_network, dry_run: this.dry_run,
             max_steps: this.max_steps, max_tool_calls: this.max_tool_calls, max_writes: this.max_writes,
             max_deletes: this.max_deletes, max_seconds: this.max_seconds,
             used: { calls: this.calls, writes: this.writes, deletes: this.deletes }, blocked: this.blocks.length };
  }

  promptBlock() {
    return `root=${this.root} | network=${this.allow_network ? 'on' : 'off'}`
      + ` | dry_run=${this.dry_run ? 'on' : 'off'}`
      + ` | denied=${this.deny.length ? this.deny.join(', ') : 'none'}`
      + ` | needs approval=${this.confirm_required.length ? this.confirm_required.join(', ') : 'none'}`
      + ` | left: ${Math.max(0, this.max_tool_calls - this.calls)} calls,`
      + ` ${Math.max(0, this.max_writes - this.writes)} writes`;
  }
}

// ---------------------------------------------------------------------------
// hooks — an event bus whose handlers can also be tools
// ---------------------------------------------------------------------------

class Hooks {
  constructor(registry = null) {
    this.hooks = [];
    this.fired = [];
    this.registry = registry || defaultRegistry;
  }

  add(event, pattern = '*', call = null, toolName = '', name = '') {
    if (!HOOK_EVENTS.includes(event)) {
      throw new ToolError(`unknown hook event '${event}'; use ${HOOK_EVENTS.join(', ')}`);
    }
    if (toolName && !this.registry.has(toolName)) throw new ToolError(`hook target tool '${toolName}' is not registered`);
    if (!call && !toolName) throw new ToolError('a hook needs either a callable or a registered tool');
    const handler = call || (async (payload) => this.registry.tools[toolName].call({ payload: payload }));
    const hook = { name: name || toolName || `${event}:${pattern}`, event, pattern, tool: toolName, call: handler };
    this.hooks.push(hook);
    return hook;
  }

  remove(name) {
    const before = this.hooks.length;
    this.hooks = this.hooks.filter((hook) => hook.name !== name);
    return before - this.hooks.length;
  }

  describe() { return this.hooks.map((h) => ({ name: h.name, event: h.event, pattern: h.pattern, tool: h.tool })); }

  async emit(event, payload) {
    /* Run matching hooks. pre_tool may block or rewrite args; others may only block. */
    const verdict = { block: false, reason: '', args: { ...(payload.args || {}) } };
    for (const hook of this.hooks) {
      if (hook.event !== event) continue;
      if (!fnmatch(String(payload.tool || ''), hook.pattern)) continue;
      let outcome;
      try { outcome = await hook.call({ ...payload }); } catch (error) {
        outcome = error instanceof Blocked ? { block: true, reason: error.message }
          : { block: true, reason: `hook ${hook.name} raised ${error.name}: ${error.message}` };
      }
      this.fired.push({ event, hook: hook.name, tool: payload.tool, outcome: outcome || {}, at: nowSeconds() });
      if (!outcome || typeof outcome !== 'object') continue;
      if (outcome.block) {
        verdict.block = true;
        verdict.reason = (verdict.reason ? `${verdict.reason}; ` : '')
          + String(outcome.reason || `blocked by ${hook.name}`);
        if (event === 'pre_tool') return verdict;
      } else if (event === 'pre_tool' && outcome.args && typeof outcome.args === 'object') {
        verdict.args = { ...verdict.args, ...outcome.args };
      }
    }
    return verdict;
  }

  register(reg) {
    reg.add(new Tool({ name: 'hook.add', desc: 'attach a hook to an event; it can block a call or rewrite its args',
                       args: { event: ['string', `one of ${HOOK_EVENTS.join(',')}`],
                               pattern: ['string', 'fnmatch on the tool name'],
                               tool: ['string', 'registered tool to run as the handler'], name: 'string' },
                       required: ['event'], kind: 'hook', mutating: true,
                       handler: async ({ event, pattern = '*', tool = '', name = '' }) => {
                         const hook = this.add(event, pattern || '*', null, tool, name);
                         return { attached: hook.name, event: hook.event, pattern: hook.pattern };
                       } }));
    reg.add(new Tool({ name: 'hook.list', desc: 'attached hooks', kind: 'hook', handler: async () => this.describe() }));
    reg.add(new Tool({ name: 'hook.remove', desc: 'detach hooks by name', args: { name: 'string' },
                       required: ['name'], kind: 'hook', mutating: true,
                       handler: async ({ name }) => this.remove(name) }));
    reg.add(new Tool({ name: 'hook.history', desc: 'recent firings with outcomes', args: { limit: 'integer' },
                       kind: 'hook', handler: async ({ limit = 20 }) => this.fired.slice(-Number(limit)) }));
  }
}

// ---------------------------------------------------------------------------
// memory — durable notes that outlive one run
// ---------------------------------------------------------------------------

class Memory {
  constructor(location = null, options = {}) {
    this.path = location ? path.resolve(String(location)) : null;
    this.clock = options.clock || nowSeconds;
    this.entries = [];
    this.load();
  }

  load() {
    if (this.path && fs.existsSync(this.path)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.path, 'utf8'));
        this.entries = Array.isArray(data) ? data : [];
      } catch { this.entries = []; }
    }
    return this.entries.length;
  }

  save() {
    if (this.path) {
      fs.mkdirSync(path.dirname(this.path), { recursive: true });
      fs.writeFileSync(this.path, JSON.stringify(this.entries, null, 2));
    }
  }

  write({ key, value = null, tags = '', kind = 'note' }) {
    const existing = this.entries.find((entry) => entry.key === key);
    const entry = { key, value, tags, kind, at: this.clock(), hits: existing ? existing.hits || 0 : 0 };
    if (existing) Object.assign(existing, entry);
    else this.entries.push(entry);
    this.save();
    return entry;
  }

  read({ key }) {
    const entry = this.entries.find((row) => row.key === key);
    if (!entry) throw new ToolError(`no memory entry '${key}'`);
    entry.hits = (entry.hits || 0) + 1;
    this.save();
    return entry;
  }

  forget({ key }) {
    const before = this.entries.length;
    this.entries = this.entries.filter((entry) => entry.key !== key);
    this.save();
    return { forgotten: before - this.entries.length, left: this.entries.length };
  }

  search({ query = '', limit = 8, kind = '' } = {}) {
    const words = String(query || '').toLowerCase().split(/[^\p{L}\p{N}_]+/u).filter(Boolean);
    const now = Math.max(1, Number(this.clock()));
    const scored = [];
    for (const entry of this.entries) {
      if (kind && entry.kind !== kind) continue;
      const haystack = `${entry.key} ${entry.tags || ''} ${entry.kind || ''} ${entry.value}`.toLowerCase();
      const score = words.length ? words.reduce((total, word) => total + haystack.split(word).length - 1, 0) : 1;
      if (score) scored.push({ entry, score: score + 0.5 * (entry.hits || 0) + Number(entry.at || 0) / now });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Number(limit)).map((row) => row.entry);
  }

  notes({ limit = 6 } = {}) {
    const recent = [...this.entries].sort((a, b) => Number(b.at || 0) - Number(a.at || 0)).slice(0, Number(limit));
    return recent.reverse().map((entry) => `- ${entry.key} (${entry.kind || 'note'}): ${String(entry.value).slice(0, 200)}`).join(NL);
  }

  register(reg) {
    reg.add(new Tool({ name: 'memory.write', desc: 'store a note under a key',
                       args: { key: 'string', value: ['any', 'what to remember'], tags: 'string',
                               kind: ['string', 'note|preference|fact|warning|chunk'] },
                       required: ['key', 'value'], kind: 'memory', mutating: true,
                       handler: async (kwargs) => this.write(kwargs) }));
    reg.add(new Tool({ name: 'memory.search', desc: 'find notes by keyword',
                       args: { query: 'string', limit: 'integer', kind: 'string' }, kind: 'memory',
                       handler: async (kwargs) => this.search(kwargs) }));
    reg.add(new Tool({ name: 'memory.read', desc: 'one note by key', args: { key: 'string' },
                       required: ['key'], kind: 'memory', handler: async (kwargs) => this.read(kwargs) }));
    reg.add(new Tool({ name: 'memory.list', desc: 'every note', kind: 'memory', handler: async () => this.entries }));
    reg.add(new Tool({ name: 'memory.forget', desc: 'drop a note', args: { key: 'string' }, required: ['key'],
                       kind: 'memory', mutating: true, handler: async (kwargs) => this.forget(kwargs) }));
  }
}

// ---------------------------------------------------------------------------
// context — the window the model actually sees
// ---------------------------------------------------------------------------

class Context {
  constructor(options = {}) {
    this.budget_tokens = Math.trunc(options.budget_tokens || 4000);
    this.keep_last = Math.trunc(options.keep_last === undefined ? 6 : options.keep_last);
    this.system = '';
    this.messages = [];
    this.compactions = 0;
  }

  reset(system = '') { this.system = system; this.messages = []; }

  push(role, content, name = '') {
    const text = typeof content === 'string' ? content : JSON.stringify(content ?? null);
    const message = { role, content: text, name, tokens: estimateTokens(text) };
    this.messages.push(message);
    return message;
  }

  stats() {
    const used = this.messages.reduce((total, message) => total + message.tokens, 0) + estimateTokens(this.system);
    return { messages: this.messages.length, tokens: used, budget: this.budget_tokens,
             used_pct: Math.round((100 * used) / Math.max(1, this.budget_tokens) * 10) / 10,
             over: used > this.budget_tokens, compactions: this.compactions };
  }

  needsCompaction() { return this.stats().over; }

  asMessages() {
    const out = this.system ? [{ role: 'system', content: this.system }] : [];
    for (const message of this.messages) {
      const entry = { role: message.role, content: message.content };
      if (message.name) entry.name = message.name;
      out.push(entry);
    }
    return out;
  }

  compact(summary = null) {
    /* Fold everything older than keep_last into one summary message. */
    const before = this.stats().tokens;
    if (this.messages.length <= this.keep_last + 1) return { compacted: false, reason: 'nothing old enough to fold', tokens: before };
    const old = this.messages.slice(0, this.messages.length - this.keep_last);
    if (summary === null || summary === undefined) {
      summary = old.map((m) => `${m.name || m.role}: ${m.content.slice(0, 70)}`).join(' | ');
    }
    const folded = [{ role: 'system', content: `[summary of ${old.length} earlier turns] ${summary}`,
                      name: 'compact', tokens: estimateTokens(summary) }];
    this.messages = [...folded, ...this.messages.slice(old.length)];
    this.compactions += 1;
    return { compacted: true, folded: old.length, before, after: this.stats().tokens };
  }

  register(reg) {
    reg.add(new Tool({ name: 'context.stats', desc: 'window usage: tokens, messages, budget', kind: 'context',
                       handler: async () => this.stats() }));
    reg.add(new Tool({ name: 'context.read', desc: 'the transcript the model would see', args: { limit: 'integer' },
                       kind: 'context', handler: async ({ limit = 30 }) => this.messages.slice(-Number(limit)) }));
    reg.add(new Tool({ name: 'context.push', desc: 'add a message to the transcript',
                       args: { role: ['string', 'user|assistant|tool'], content: 'string', name: 'string' },
                       required: ['role', 'content'], kind: 'context', mutating: true,
                       handler: async ({ role, content, name = '' }) => this.push(role, content, name) }));
  }
}

// ---------------------------------------------------------------------------
// prompts — layered system prompt assembly
// ---------------------------------------------------------------------------

const BASE_PROMPT = `You are the {name} harness agent, a loop that acts only through tools.

Reply with ONE JSON object and nothing else:
  {{"thought": "<one line>", "call": {{"tool": "<name>", "args": {{...}}}}}}
  {{"thought": "<one line>", "final": "<answer to the goal>"}}

Use only tools from the list. Read before you write. If a call is blocked by a
guardrail, do not fight it: route around it or explain the block in "final".`;

const PERSONA_PROMPT = 'Style: terse and concrete. Never invent a path or a tool name. Quote the output that decided something.';

class Prompts {
  constructor(layers = null) {
    this.layers = { base: BASE_PROMPT, persona: PERSONA_PROMPT, goal: '', guardrails: '', skills: '', memory: '', tools: '' };
    this.dynamic = [];
    Object.assign(this.layers, layers || {});
  }

  set({ layer, text }) {
    if (!(layer in this.layers)) throw new ToolError(`unknown prompt layer '${layer}'; layers are ${Object.keys(this.layers).join(', ')}`);
    this.layers[layer] = text;
    return { layer, tokens: estimateTokens(text) };
  }

  attach(layer, producer) {
    /* A layer recomputed every turn, e.g. ("tasks", () => JSON.stringify(tasks.summary())). */
    if (!(layer in this.layers)) this.layers[layer] = '';
    this.dynamic.push([layer, producer]);
    return { layer, dynamic: true };
  }

  render(agent) {
    for (const [layer, producer] of this.dynamic) this.layers[layer] = String(producer() || '');
    this.layers.memory = agent.memory.notes();
    this.layers.skills = agent.skills.promptBlock();
    this.layers.guardrails = agent.policy.promptBlock();
    this.layers.tools = agent.toolPrompt();
    const order = ['base', 'persona', 'goal', 'guardrails', 'skills', 'memory', 'tools']
      .concat(Object.keys(this.layers).filter((key) => !['base', 'persona', 'goal', 'guardrails', 'skills', 'memory', 'tools'].includes(key)));
    const parts = [this.layers.base.replaceAll('{name}', agent.name)];
    for (const key of order.slice(1)) {
      const text = this.layers[key];
      if (text) parts.push(key === 'persona' ? text : `# ${key}\n${text}`);
    }
    return parts.filter((part) => String(part).trim()).join('\n\n');
  }

  register(reg, agent) {
    const get = async ({ rendered = false } = {}) => {
      const out = { layers: Object.fromEntries(Object.entries(this.layers)
        .map(([key, value]) => [key, String(value).length > 400 ? `${String(value).slice(0, 400)}...` : value])),
                   dynamic: this.dynamic.map(([layer]) => layer) };
      if (rendered) { out.rendered = this.render(agent); out.tokens = estimateTokens(out.rendered); }
      return out;
    };
    reg.add(new Tool({ name: 'prompt.get', desc: 'the prompt layers, and the rendered system prompt on request',
                       args: { rendered: ['boolean', 'include the assembled text'] }, kind: 'prompt', handler: get }));
    reg.add(new Tool({ name: 'prompt.set', desc: 'replace one layer (base|persona|goal|guardrails|skills|memory|tools)',
                       args: { layer: 'string', text: 'string' }, required: ['layer', 'text'], kind: 'prompt',
                       mutating: true, handler: async (kwargs) => this.set(kwargs) }));
  }
}

// ---------------------------------------------------------------------------
// skills — instruction packs read from markdown
// ---------------------------------------------------------------------------

class Skills {
  constructor(dirs = ['skills'], options = {}) {
    this.dirs = (Array.isArray(dirs) ? dirs : [dirs]).map((dir) => path.resolve(String(dir)));
    this.registry = options.registry || defaultRegistry;
    this.packs = {};
    this.active = [];
    this.scan();
  }

  scan() {
    const found = [];
    for (const directory of this.dirs) {
      let listing;
      try { listing = fs.readdirSync(directory, { withFileTypes: true }); } catch { continue; }
      const files = [];
      for (const entry of listing) {
        if (entry.isFile() && entry.name.endsWith('.md')) files.push(path.join(directory, entry.name));
        else if (entry.isDirectory()) {
          for (const nested of fs.readdirSync(path.join(directory, entry.name), { withFileTypes: true })) {
            if (nested.isFile() && nested.name.endsWith('.md')) files.push(path.join(directory, entry.name, nested.name));
          }
        }
      }
      for (const file of files.sort()) {
        const pack = this.parse(file);
        if (pack && pack.name) { this.packs[pack.name] = pack; found.push(pack.name); }
      }
    }
    return found;
  }

  parse(file) {
    let raw;
    try { raw = fs.readFileSync(file, 'utf8'); } catch { return null; }
    const match = raw.match(/^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n([\s\S]*)$/);
    const meta = match ? match[1] : '';
    const body = (match ? match[2] : raw).trim();
    const fields = {};
    for (const line of meta.split(/\r?\n/)) {
      const index = line.indexOf(':');
      if (index === -1) continue;
      fields[line.slice(0, index).trim()] = line.slice(index + 1).trim().replace(/^["']|["']$/g, '');
    }
    return { name: fields.name || path.basename(file, '.md'), description: fields.description || '',
             when: fields.when_to_use || fields.when || '',
             tools: (fields.tools || '').split(',').map((item) => item.trim()).filter(Boolean),
             body, file };
  }

  list() {
    return Object.values(this.packs).map((pack) => ({ name: pack.name, description: pack.description, when: pack.when,
                                                     tools: pack.tools, active: this.active.includes(pack.name) }));
  }

  pack(name) {
    if (!(name in this.packs)) {
      throw new ToolError(`no skill '${name}'; ${Object.keys(this.packs).length} installed: `
        + `${Object.keys(this.packs).join(', ') || 'none'}`);
    }
    return this.packs[name];
  }

  use({ name }) {
    const pack = this.pack(name);
    if (!this.active.includes(name)) this.active.push(name);
    return { active: [...this.active], requires: pack.tools,
             missing_tools: pack.tools.filter((toolName) => !this.registry.has(toolName)) };
  }

  stop({ name }) {
    this.active = this.active.filter((item) => item !== name);
    return { active: [...this.active] };
  }

  promptBlock() {
    return this.active.filter((name) => name in this.packs)
      .map((name) => `## skill: ${name}\n${this.packs[name].body}`).join('\n\n');
  }

  register(reg) {
    reg.add(new Tool({ name: 'skills.list', desc: 'installed skill packs', kind: 'skill',
                       handler: async () => this.list() }));
    reg.add(new Tool({ name: 'skills.read', desc: "a skill's full instructions", args: { name: 'string' },
                       required: ['name'], kind: 'skill', handler: async ({ name }) => this.pack(name).body }));
    reg.add(new Tool({ name: 'skills.use', desc: 'activate a skill; its body joins the system prompt',
                       args: { name: 'string' }, required: ['name'], kind: 'skill', mutating: true,
                       handler: async (kwargs) => this.use(kwargs) }));
    reg.add(new Tool({ name: 'skills.stop', desc: 'deactivate a skill', args: { name: 'string' }, required: ['name'],
                       kind: 'skill', mutating: true, handler: async (kwargs) => this.stop(kwargs) }));
    reg.add(new Tool({ name: 'skills.scan', desc: 're-read the skill directories', kind: 'skill', mutating: true,
                       handler: async () => ({ found: this.scan() }) }));
  }
}

// ---------------------------------------------------------------------------
// tasks and scheduler
// ---------------------------------------------------------------------------

class Tasks {
  constructor(options = {}) {
    this.items = [];
    this.clock = options.clock || nowSeconds;
    this.next = 1;
  }

  create({ subject, description = '', status = 'pending', blocks = '' }) {
    const task = { id: this.next, subject, description, status,
                   blocked_by: (String(blocks).match(/\d+/g) || []).map(Number), at: this.clock(), history: [status] };
    this.next += 1;
    this.items.push(task);
    return task;
  }

  get({ id }) {
    const task = this.items.find((item) => item.id === Number(id));
    if (!task) throw new ToolError(`no task ${id}`);
    return task;
  }

  update({ id, status = '', subject = '', description = '' }) {
    const task = this.get({ id });
    if (status) {
      if (!TASK_STATUSES.includes(status)) throw new ToolError(`bad task status '${status}'; use ${TASK_STATUSES.join(', ')}`);
      task.status = status;
      task.history.push(status);
    }
    if (subject) task.subject = subject;
    if (description) task.description = description;
    return task;
  }

  ready() {
    const done = new Set(this.items.filter((task) => task.status === 'completed').map((task) => task.id));
    return this.items.filter((task) => task.status === 'pending' && task.blocked_by.every((id) => done.has(id)));
  }

  summary() {
    const counts = {};
    for (const task of this.items) counts[task.status] = (counts[task.status] || 0) + 1;
    return { tasks: this.items.length, ...counts, ready: this.ready().map((task) => task.subject) };
  }

  register(reg) {
    reg.add(new Tool({ name: 'task.create', desc: 'track a unit of work',
                       args: { subject: 'string', description: 'string', status: 'string',
                               blocks: ['string', 'comma separated task ids this waits on'] },
                       required: ['subject'], kind: 'task', mutating: true,
                       handler: async (kwargs) => this.create(kwargs) }));
    reg.add(new Tool({ name: 'task.list', desc: 'every task plus status counts', kind: 'task',
                       handler: async () => ({ items: this.items, ...this.summary() }) }));
    reg.add(new Tool({ name: 'task.get', desc: 'one task', args: { id: 'integer' }, required: ['id'], kind: 'task',
                       handler: async (kwargs) => this.get(kwargs) }));
    reg.add(new Tool({ name: 'task.update', desc: "change a task's status or text",
                       args: { id: 'integer', status: 'string', subject: 'string', description: 'string' },
                       required: ['id'], kind: 'task', mutating: true, handler: async (kwargs) => this.update(kwargs) }));
    reg.add(new Tool({ name: 'task.ready', desc: 'tasks with no unfinished blockers', kind: 'task',
                       handler: async () => this.ready() }));
  }
}

class Scheduler {
  /* Interval/at jobs. A job is a tool call through the same funnel — nothing else. */
  constructor(options = {}) {
    this.clock = options.clock || nowSeconds;
    this.jobs = [];
    this.fired = [];
  }

  add({ name, tool: toolName = '', args = null, every = 0, at = 0, max_runs: maxRuns = 0 }) {
    if (!toolName) throw new ToolError('a scheduled job must name a tool to call');
    if (!every && !at) throw new ToolError('give either every=<seconds> or at=<epoch>');
    const job = { name, tool: toolName, args: args || {}, every: Number(every || 0), at: Number(at || 0),
                  max_runs: Math.trunc(maxRuns || 0), runs: 0, last: 0,
                  next: at ? Number(at) : this.clock() + Number(every) };
    this.jobs = this.jobs.filter((row) => row.name !== name).concat([job]);
    return job;
  }

  due(now = null) {
    const at = now === null ? Number(this.clock()) : Number(now);
    return this.jobs.filter((job) => job.next && job.runs < (job.max_runs || Number.MAX_SAFE_INTEGER) && job.next <= at);
  }

  async tick(dispatch, now = null) {
    /* Fire each due job once. dispatch(toolName, args) returns a result. */
    const at = now === null ? Number(this.clock()) : Number(now);
    const results = [];
    for (const job of this.due(at)) {
      const outcome = { job: job.name, tool: job.tool, at };
      try { outcome.result = await dispatch(job.tool, job.args); }
      catch (error) { outcome.error = `${error.name}: ${error.message}`; }
      job.runs += 1;
      job.last = at;
      job.next = job.every ? at + job.every : 0;
      this.fired.push(outcome);
      results.push(outcome);
    }
    return results;
  }

  remove({ name }) {
    const before = this.jobs.length;
    this.jobs = this.jobs.filter((job) => job.name !== name);
    return { removed: before - this.jobs.length, left: this.jobs.length };
  }

  register(reg) {
    reg.add(new Tool({ name: 'schedule.add', desc: 'run a tool every N seconds or at an epoch time',
                       args: { name: 'string', tool: 'string', args: 'object',
                               every: ['number', 'seconds between runs'], at: ['number', 'epoch fire time'],
                               max_runs: 'integer' },
                       required: ['name', 'tool'], kind: 'schedule', mutating: true,
                       handler: async (kwargs) => this.add(kwargs) }));
    reg.add(new Tool({ name: 'schedule.list', desc: 'jobs, their next run, and recent firings', kind: 'schedule',
                       handler: async () => ({ jobs: this.jobs, fired: this.fired.slice(-10) }) }));
    reg.add(new Tool({ name: 'schedule.due', desc: 'jobs whose time has come', kind: 'schedule',
                       handler: async () => ({ due: this.due().map((job) => job.name) }) }));
    reg.add(new Tool({ name: 'schedule.remove', desc: 'drop a job', args: { name: 'string' }, required: ['name'],
                       kind: 'schedule', mutating: true, handler: async (kwargs) => this.remove(kwargs) }));
  }
}

// ---------------------------------------------------------------------------
// models — every endpoint is a tool with one action contract
// ---------------------------------------------------------------------------

async function httpJson(url, payload, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), (options.timeout || 30) * 1000);
  let text, status;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(options.headers || {}) },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    text = await response.text();
    status = response.status;
    if (!response.ok) throw new ToolError(`${url} -> HTTP ${status}: ${text.slice(0, 300)}`);
  } catch (error) {
    if (error instanceof ToolError) { clearTimeout(timer); throw error; }
    clearTimeout(timer);
    throw new ToolError(`${url} unreachable: ${error.message}`);
  }
  clearTimeout(timer);
  try { return JSON.parse(text); } catch { throw new ToolError(`${url} returned non-JSON: ${text.slice(0, 200)}`); }
}

function splitCommand(cmd) {
  /* shlex.split for the common case: words, single and double quotes, backslash escapes. */
  const argv = [];
  let current = '', quoted = null, started = false;
  for (let index = 0; index < String(cmd).length; index += 1) {
    const char = String(cmd)[index];
    if (quoted) {
      if (char === quoted) quoted = null;
      else if (char === '\\' && quoted === '"') { index += 1; current += String(cmd)[index] || ''; }
      else current += char;
    } else if (char === '"' || char === "'") { quoted = char; started = true; }
    else if (/\s/.test(char)) { if (started) { argv.push(current); current = ''; started = false; } }
    else { current += char; started = true; }
  }
  if (started) argv.push(current);
  return argv;
}

const chatmlToken = (label) => String.fromCharCode(60) + '|' + label + '|' + String.fromCharCode(62);
const CHATML_START = chatmlToken('start');
const CHATML_MESSAGE = chatmlToken('message');
const CHATML_END = chatmlToken('end');
// both turn terminators are stop tokens: a model emits whichever its own template used
const CHATML_STOPS = [CHATML_END, chatmlToken('im_end')];

/* llama.cpp /completion takes a raw prompt; most GGUFs are trained on ChatML.
   llama.cpp's own template is START+role+MESSAGE+content+END per turn, then an open
   START+assistant+MESSAGE for the reply. chatmlToken keeps the control sequences out of
   the source so this file stays safe to paste into a chat. */
const renderChatml = (messages) => messages.map((m) => `${CHATML_START}${m.role || 'user'}${CHATML_MESSAGE}${m.content || ''}${CHATML_END}`).join('')
  + `${CHATML_START}assistant${CHATML_MESSAGE}`;

class Models {
  /* The brain is pluggable and callable: model.* are ordinary tools. */
  constructor(options = {}) {
    this.registry = options.registry || defaultRegistry;
    this.env = options.env || process.env;
    this.scripts = {};
    this.turns = {};
    this.log = [];
    if (options.scripted) this.scripts.default = [...options.scripted];
  }

  timeoutFor(fallback) { return Number(this.env.AGENT_TIMEOUT || fallback); }

  script({ script = 'default', turns = null, append = false }) {
    const rows = typeof turns === 'string' ? JSON.parse(turns) : (turns || []);
    this.scripts[script] = append ? [...(this.scripts[script] || []), ...rows] : [...rows];
    delete this.turns[script];
    return { script, turns: this.scripts[script].length };
  }

  scripted({ messages = null, tools = null, goal = '', step = 0, script = 'default' } = {}) {
    /* Offline, deterministic, and the reason every example runs with no key. */
    const queue = this.scripts[script] || (this.scripts[script] = []);
    const index = this.turns[script] || 0;
    this.turns[script] = index + 1;
    this.log.push({ model: 'scripted', script, step, tools: (tools || []).length });
    if (index < queue.length) return { ...queue[index] };
    return { thought: 'script exhausted', final: `(script '${script}' ran out after ${index} turns; goal=${String(goal).slice(0, 60)})` };
  }

  async ollama({ messages = null, tools = null, goal = '', step = 0, model = '', url = '', format = 'json', temperature = 0.2 } = {}) {
    /* POST /api/chat on Ollama. format=json keeps small local models on contract. */
    const base = url || this.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
    const payload = { model: model || this.env.OLLAMA_MODEL || 'qwen3:8b', messages: messages || [], stream: false, format,
                      options: { temperature: Number(temperature), num_ctx: 8192 } };
    if (tools && tools.length) payload.tools = tools.filter((name) => this.registry.has(name)).map((name) => this.registry.tools[name].openaiSpec());
    const data = await httpJson(`${base.replace(/\/+$/, '')}/api/chat`, payload, { timeout: this.timeoutFor(120) });
    const content = (data.message || {}).content || data.response || '';
    return this.actionFrom(content, (data.message || {}).tool_calls);
  }

  async openai({ messages = null, tools = null, goal = '', step = 0, model = '', base_url: baseUrl = '', api_key: apiKey = '' } = {}) {
    /* POST /chat/completions on any OpenAI-compatible server (vLLM, LM Studio, llama.cpp, OpenAI). */
    const base = baseUrl || this.env.OPENAI_BASE_URL || 'http://127.0.0.1:11434/v1';
    const key = apiKey || this.env.OPENAI_API_KEY || 'ollama';
    const specs = (tools || []).filter((name) => this.registry.has(name)).map((name) => this.registry.tools[name].openaiSpec());
    const payload = { model: model || this.env.OPENAI_MODEL || 'qwen3:8b', messages: messages || [], temperature: 0.2, tool_choice: 'auto' };
    if (specs.length) payload.tools = specs;
    const data = await httpJson(`${base.replace(/\/+$/, '')}/chat/completions`, payload,
                                { timeout: this.timeoutFor(120), headers: { authorization: `Bearer ${key}` } });
    const choice = ((data.choices || [])[0] || {}).message;
    if (!choice) throw new ToolError(`unexpected reply: ${JSON.stringify(data).slice(0, 300)}`);
    return this.actionFrom(choice.content || '', choice.tool_calls);
  }

  async llamacpp({ messages = null, tools = null, goal = '', step = 0, base_url: baseUrl = '', model = '',
                   temperature = 0.1, n_predict: nPredict = 400 } = {}) {
    /* POST /completion on a llama.cpp server, forcing the action shape with a json_schema. */
    const base = baseUrl || this.env.LLAMACPP_URL || 'http://127.0.0.1:8080';
    const toolList = (tools || []).filter((name) => this.registry.has(name))
      .map((name) => `- ${name}: ${this.registry.tools[name].desc}`).join(NL);
    const prompt = renderChatml([{ role: 'system', content: (messages || [{}])[0].content || '' },
                                 { role: 'user', content: `Tools:${NL}${toolList}${NL}Goal: ${goal}` }]);
    const payload = { prompt, temperature: Number(temperature), n_predict: Math.trunc(nPredict), stream: false,
                      no_generation: false, stop: CHATML_STOPS,
                      json_schema: JSON.stringify({ type: 'object', properties: {
                        thought: { type: 'string' },
                        call: { type: 'object', properties: { tool: { enum: tools || [] }, args: { type: 'object' } } },
                        final: { type: 'string' } }, required: ['thought'] }) };
    // a router-mode llama-server hosts several models and cannot guess which one
    if (model) payload.model = model;
    const data = await httpJson(`${base.replace(/\/+$/, '')}/completion`, payload, { timeout: this.timeoutFor(180) });
    let raw = data.content || data.generation || '';
    for (const stop of CHATML_STOPS) raw = raw.split(stop)[0];
    return this.actionFrom(raw);
  }

  actionFrom(content, toolCalls = null) {
    /* Normalise whatever the server said into {thought, call} or {thought, final}. */
    if (toolCalls && toolCalls.length) {
      const call = toolCalls[0].function || {};
      const args = typeof call.arguments === 'string' ? JSON.parse(call.arguments) : (call.arguments || {});
      return { thought: content || call.name || '', call: { tool: call.name || '', args } };
    }
    const action = firstJson(content);
    if (!action) throw new ToolError(`model did not return the action contract: ${JSON.stringify(String(content).slice(0, 200))}`);
    if ('action' in action && !('call' in action)) { action.call = action.action; delete action.action; }
    return action;
  }

  register(reg) {
    const common = { messages: ['array', 'transcript'], tools: ['array', 'tool names the model may call'],
                     goal: ['string', 'what the run is for'], step: ['integer', 'turn index'] };
    reg.add(new Tool({ name: 'model.scripted', desc: 'offline deterministic model: replays a queued script of actions',
                       args: { ...common, script: ['string', 'named script queue'] }, kind: 'model',
                       handler: async (kwargs) => this.scripted(kwargs) }));
    reg.add(new Tool({ name: 'model.set_script', desc: 'queue the actions model.scripted should return',
                       args: { script: 'string', turns: 'array', append: ['boolean', 'add to the queue instead of replacing'] },
                       required: ['turns'], kind: 'model', mutating: true, handler: async (kwargs) => this.script(kwargs) }));
    reg.add(new Tool({ name: 'model.ollama', desc: 'Ollama /api/chat as the decision maker',
                       args: { ...common, model: 'string', url: 'string', format: 'string', temperature: 'number' },
                       kind: 'model', network: true, handler: (kwargs) => this.ollama(kwargs) }));
    reg.add(new Tool({ name: 'model.openai', desc: 'any OpenAI-compatible /chat/completions as the decision maker',
                       args: { ...common, model: 'string', base_url: 'string', api_key: 'string' },
                       kind: 'model', network: true, handler: (kwargs) => this.openai(kwargs) }));
    reg.add(new Tool({ name: 'model.llamacpp', desc: 'llama.cpp server /completion with json_schema as the decision maker',
                       args: { ...common, base_url: 'string', model: 'string', temperature: 'number', n_predict: 'integer' },
                       kind: 'model', network: true, handler: (kwargs) => this.llamacpp(kwargs) }));
    reg.add(new Tool({ name: 'model.log', desc: 'recent model calls: which endpoint, how many tools offered',
                       args: { limit: 'integer' }, kind: 'model',
                       handler: async ({ limit = 20 }) => this.log.slice(-Number(limit)) }));
  }
}

// ---------------------------------------------------------------------------
// built-in tools — the small pack every harness needs, still behind the funnel
// ---------------------------------------------------------------------------

class Builtins {
  constructor(root, options = {}) {
    this.root = path.resolve(String(root));
    this.clock = options.clock || nowSeconds;
  }

  targetPath(value) { return path.resolve(resolvePath(value, this.root)); }

  async fetchPage(url, timeout = 20, maxBytes = 20000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Number(timeout) * 1000);
    try {
      const response = await fetch(url, { signal: controller.signal });
      return { url, status: response.status, body: (await response.text()).slice(0, Math.trunc(maxBytes)) };
    } catch (error) {
      throw new ToolError(`${url} unreachable: ${error.message}`);
    } finally { clearTimeout(timer); }
  }

  register(reg) {
    const root = this.root;
    reg.add(new Tool({ name: 'fs.read', desc: 'read a text file inside the sandbox',
                       args: { path: 'string', max_bytes: 'integer' }, required: ['path'], kind: 'tool',
                       plugin: 'builtins',
                       handler: async ({ path: wanted, max_bytes: maxBytes = 20000 }) => {
                         const target = this.targetPath(wanted);
                         if (!fs.existsSync(target) || !fs.statSync(target).isFile()) {
                           throw new ToolError(`${wanted} is not a file under ${root}`);
                         }
                         const all = fs.readFileSync(target);
                         return { path: target, bytes: all.length,
                                  content: all.subarray(0, Math.trunc(maxBytes)).toString('utf8') };
                       } }));
    reg.add(new Tool({ name: 'fs.write', desc: 'write or append a file inside the sandbox',
                       args: { path: 'string', content: 'string', append: 'boolean' }, required: ['path'],
                       kind: 'tool', mutating: true, plugin: 'builtins',
                       handler: async ({ path: wanted, content = '', append = false }) => {
                         const target = this.targetPath(wanted);
                         fs.mkdirSync(path.dirname(target), { recursive: true });
                         const text = String(content);
                         if (append && fs.existsSync(target)) fs.appendFileSync(target, `\n${text}`);
                         else fs.writeFileSync(target, text);
                         return { path: target, bytes: fs.statSync(target).size, mode: append ? 'append' : 'write' };
                       } }));
    reg.add(new Tool({ name: 'fs.edit', desc: 'replace text inside a file; a second match needs all=true',
                       args: { path: 'string', find: 'string', replace: 'string', all: 'boolean' },
                       required: ['path', 'find', 'replace'], kind: 'tool', mutating: true, plugin: 'builtins',
                       handler: async ({ path: wanted, find, replace = '', all = false }) => {
                         const target = this.targetPath(wanted);
                         if (!fs.existsSync(target) || !fs.statSync(target).isFile()) {
                           throw new ToolError(`${wanted} is not a file under ${root}`);
                         }
                         const content = fs.readFileSync(target, 'utf8');
                         const needle = String(find);
                         const hits = content.split(needle).length - 1;
                         if (!hits) throw new ToolError(`no match for ${needle.slice(0, 40)} in ${wanted}`);
                         if (!all && hits > 1) {
                           throw new ToolError(`${needle.slice(0, 40)} matches ${hits} times in ${wanted};`
                             + ' widen it, or pass all=true');
                         }
                         const next = all ? content.split(needle).join(String(replace))
                                          : content.replace(needle, () => String(replace));
                         fs.writeFileSync(target, next);
                         return { path: target, bytes: fs.statSync(target).size,
                                  replacements: all ? hits : 1, matches: hits };
                       } }));
    reg.add(new Tool({ name: 'fs.list', desc: 'list a directory', args: { path: 'string', pattern: 'string' },
                       kind: 'tool', plugin: 'builtins',
                       handler: async ({ path: wanted = '.', pattern = '*' }) => {
                         const target = this.targetPath(wanted);
                         if (!fs.existsSync(target) || !fs.statSync(target).isDirectory()) throw new ToolError(`${wanted} is not a directory`);
                         const matcher = pattern === '*' ? null : globToRegExp(pattern);
                         const rows = fs.readdirSync(target).filter((name) => !matcher || matcher.test(name)).sort().slice(0, 200);
                         return { path: target, entries: rows.map((name) => {
                           const stat = fs.statSync(path.join(target, name));
                           return { name, dir: stat.isDirectory(), bytes: stat.isFile() ? stat.size : 0 };
                         }) };
                       } }));
    reg.add(new Tool({ name: 'fs.exists', desc: 'does this path exist?', args: { path: 'string' }, required: ['path'],
                       kind: 'tool', plugin: 'builtins',
                       handler: async ({ path: wanted }) => {
                         const target = this.targetPath(wanted);
                         let exists = false, dir = false, file = false;
                         try { const stat = fs.statSync(target); exists = true; dir = stat.isDirectory(); file = stat.isFile(); } catch { /* absent */ }
                         return { path: target, exists, dir, file };
                       } }));
    reg.add(new Tool({ name: 'fs.delete', desc: 'remove one file (irreversible)', args: { path: 'string' },
                       required: ['path'], kind: 'tool', mutating: true, danger: true, deletes: true, plugin: 'builtins',
                       handler: async ({ path: wanted }) => {
                         const target = this.targetPath(wanted);
                         if (fs.existsSync(target) && fs.statSync(target).isDirectory()) {
                           throw new ToolError('fs.delete removes files only; a directory is too blunt an instrument');
                         }
                         if (!fs.existsSync(target)) throw new ToolError(`${wanted} does not exist`);
                         fs.unlinkSync(target);
                         return { deleted: target };
                       } }));
    reg.add(new Tool({ name: 'shell.run', desc: 'run a program with argv; no shell interpolation',
                       args: { cmd: 'string', timeout: 'number' }, required: ['cmd'], kind: 'tool',
                       mutating: true, danger: true, plugin: 'builtins',
                       handler: async ({ cmd, timeout = 20 }) => {
                         const argv = splitCommand(cmd);
                         if (!argv.length) throw new ToolError('empty command');
                         const done = spawnSync(argv[0], argv.slice(1), { cwd: root, timeout: Number(timeout) * 1000,
                                                                          encoding: 'utf8', windowsHide: true });
                         if (done.error) {
                           if (done.error.code === 'ENOENT') throw new ToolError(`program not found: ${argv[0]}`);
                           if (done.error.code === 'ETIMEDOUT') throw new ToolError(`${cmd} timed out after ${timeout}s`);
                           throw new ToolError(`${cmd} failed: ${done.error.message}`);
                         }
                         return { cmd, code: done.status, stdout: (done.stdout || '').slice(0, 8000),
                                  stderr: (done.stderr || '').slice(0, 4000) };
                       } }));
    reg.add(new Tool({ name: 'http.get', desc: 'fetch a URL', args: { url: 'string', timeout: 'number', max_bytes: 'integer' },
                       required: ['url'], kind: 'tool', network: true, plugin: 'builtins',
                       handler: async ({ url, timeout = 20, max_bytes: maxBytes = 20000 }) => this.fetchPage(url, timeout, maxBytes) }));
    reg.add(new Tool({ name: 'browser.open', desc: 'fetch a URL and reduce it to title, text and links',
                       args: { url: 'string', timeout: 'number', max_bytes: 'integer' }, required: ['url'],
                       kind: 'tool', network: true, plugin: 'builtins',
                       handler: async ({ url, timeout = 20, max_bytes: maxBytes = 40000 }) => {
                         const page = await this.fetchPage(url, timeout, maxBytes);
                         return { url: page.url, status: page.status, ...htmlReduce(page.body) };
                       } }));
    reg.add(new Tool({ name: 'time.now', desc: 'current epoch seconds', kind: 'tool', plugin: 'builtins',
                       handler: async () => ({ now: this.clock() }) }));
    reg.add(new Tool({ name: 'echo', desc: 'return its argument - the hello-world plug', args: { text: 'string' },
                       required: ['text'], kind: 'tool', plugin: 'builtins',
                       handler: async ({ text }) => ({ echo: text }) }));
  }
}

// ---------------------------------------------------------------------------
// the loop — one funnel, repeated: registry -> hooks -> policy -> handler -> hooks -> context
// ---------------------------------------------------------------------------

class Call {
  constructor(fields = {}) {
    Object.assign(this, { tool: '', kind: 'tool', plugin: 'core', args: {}, ok: false, value: null, error: '',
                          blocked: false, reason: '', source: 'loop', ms: 0 }, fields);
  }

  observation() {
    if (this.blocked) return { blocked: true, tool: this.tool, reason: this.reason };
    if (!this.ok) return { error: this.error, tool: this.tool };
    return { tool: this.tool, result: this.value };
  }

  brief() { return { ...this }; }
}

class Turn {
  constructor(i, fields = {}) {
    Object.assign(this, { i, thought: '', action: {}, call: null, final: '' }, fields);
  }

  brief() {
    return { i: this.i, thought: this.thought, tool: this.call ? this.call.tool : '',
             ok: this.call ? this.call.ok : null, blocked: this.call ? this.call.blocked : null,
             final: this.final, action: this.action };
  }
}

class Run {
  constructor(goal, model, fields = {}) {
    Object.assign(this, { goal, model, status: 'running', turns: [], started: 0, seconds: 0,
                          memory_writes: 0, tasks_created: 0, compactions: 0, error: '' }, fields);
  }

  get trail() { return this.turns.filter((turn) => turn.call).map((turn) => turn.call.tool).join(' -> '); }

  get blocked() {
    return this.turns.filter((turn) => turn.call && turn.call.blocked)
      .map((turn) => ({ i: turn.i, tool: turn.call.tool, reason: turn.call.reason }));
  }

  get answer() {
    for (let index = this.turns.length - 1; index >= 0; index -= 1) {
      if (this.turns[index].final) return this.turns[index].final;
    }
    return '';
  }

  get ok() { return this.status === 'done'; }

  brief() {
    return { status: this.status, goal: this.goal, model: this.model, answer: this.answer, trail: this.trail,
             turns: this.turns.length, blocked: this.blocked, seconds: this.seconds };
  }

  toString() { return `[${this.model}] ${this.status}: ${(this.answer || this.trail).slice(0, 120)}`; }
}

class Agent {
  /* A wired harness: nine components, one registry, one funnel, one loop. */
  constructor(options = {}) {
    const {
      name = 'harness', root = '.', model = 'model.scripted', tools_dir: toolsDir = null,
      skills_dir: skillsDir = ['skills'], memory_path: memoryPath = null, budget_tokens: budgetTokens = 4000,
      keep_last: keepLast = 6, max_steps: maxSteps = 12, allow_network: allowNetwork = false,
      dry_run: dryRun = false, extra_dirs: extraDirs = [], registry = null, env = null,
      clock = nowSeconds, verbose = false,
    } = options;
    this.name = name;
    this.root = path.resolve(String(root));
    this.clock = clock;
    // start from a copy of the process-wide registry so tool() plugins are adopted,
    // then keep our own so agents cannot leak into each other.
    this.registry = registry || new Registry();
    Object.assign(this.registry.tools, defaultRegistry.tools);
    this.components = {};

    this.policy = new Policy({ root: this.root, allow_network: allowNetwork, dry_run: dryRun,
                               max_steps: maxSteps, extra_dirs: extraDirs });
    this.hooks = new Hooks(this.registry);
    this.memory = new Memory(memoryPath, { clock });
    this.context = new Context({ budget_tokens: budgetTokens, keep_last: keepLast });
    this.prompts = new Prompts();
    this.skills = new Skills(skillsDir === null ? [] : skillsDir, { registry: this.registry });
    this.tasks = new Tasks({ clock });
    this.scheduler = new Scheduler({ clock });
    this.models = new Models({ registry: this.registry, env: env || process.env });
    this.builtins = new Builtins(this.root, { clock });
    for (const component of [this.policy, this.hooks, this.memory, this.context, this.prompts, this.skills,
                             this.tasks, this.scheduler, this.models, this.builtins]) {
      if (!(component.constructor.name in this.components)) this.components[component.constructor.name] = component;
    }
    this.model = model;
    this.tool_defaults = {};          // operator-set args merged into every call
    this.journal = [];
    this.runs = [];
    this.verbose = verbose;
    this.tools_dir = toolsDir;
    this.plugins = toolsDir ? discover(toolsDir, { registry: this.registry })
                            : { loaded: [], modules: [], errors: {} };
    this.registerLoopTools();
  }

  // -- wiring ------------------------------------------------------------
  registerLoopTools() {
    const reg = this.registry;
    this.hooks.register(reg);
    this.memory.register(reg);
    this.context.register(reg);
    this.prompts.register(reg, this);
    this.skills.register(reg);
    this.tasks.register(reg);
    this.scheduler.register(reg);
    this.models.register(reg);
    this.builtins.register(reg);

    reg.add(new Tool({ name: 'tools.list', desc: 'every registered tool, by component kind',
                       args: { kind: ['string', 'tool|skill|memory|context|hook|policy|prompt|loop|task|schedule|model'] },
                       kind: 'loop',
                       handler: async ({ kind = '' } = {}) => ({ count: this.registry.length,
                         kinds: Object.fromEntries(Object.values(this.registry.tools).reduce((map, spec) => {
                           map.set(spec.kind, (map.get(spec.kind) || 0) + 1); return map; }, new Map())),
                         tools: this.registry.brief(kind || null) }) }));
    reg.add(new Tool({ name: 'tools.describe', desc: 'one tool: its description, args and guardrail flags',
                       args: { tool: 'string' }, required: ['tool'], kind: 'loop',
                       handler: async ({ tool: name }) => this.registry.get(name).brief() }));
    reg.add(new Tool({ name: 'tools.call', desc: 'invoke any registered tool by name - the meta plug',
                       args: { tool: 'string', args: 'object' }, required: ['tool'], kind: 'loop', mutating: true,
                       handler: async ({ tool: name, args = null }) => (await this.call(name, args || {}, 'tools.call')).observation() }));
    reg.add(new Tool({ name: 'policy.get', desc: 'current guardrail settings and budget usage', kind: 'policy',
                       handler: async () => this.policy.summary() }));
    reg.add(new Tool({ name: 'policy.set', desc: 'change guardrails: allow, deny, dry_run, allow_network, max_* ...',
                       args: { allow: 'array', deny: 'array', confirm_required: 'array', extra_dirs: 'array',
                               max_steps: 'integer', max_tool_calls: 'integer', max_writes: 'integer',
                               max_deletes: 'integer', max_seconds: 'number', allow_network: 'boolean', dry_run: 'boolean' },
                       kind: 'policy', mutating: true, handler: async (changes) => this.policy.set(this.cleanPolicyChanges(changes)) }));
    reg.add(new Tool({ name: 'policy.check', desc: 'ask whether the guardrails would allow a call',
                       args: { tool: 'string', args: 'object' }, required: ['tool'], kind: 'policy',
                       handler: async ({ tool: name, args = null }) => {
                         const spec = this.registry.get(name);
                         const snapshot = [this.policy.calls, this.policy.writes, this.policy.deletes, this.policy.blocks.length];
                         const [ok, reason] = this.policy.check(spec, args || {});
                         [this.policy.calls, this.policy.writes, this.policy.deletes] = snapshot;
                         this.policy.blocks.length = snapshot[3];
                         return { allowed: ok, reason, tool: name, kind: spec.kind,
                                  mutating: spec.mutating, network: spec.network, danger: spec.danger };
                       } }));
    reg.add(new Tool({ name: 'confirm.grant', desc: 'operator approval for one confirm-required tool (one call unless repeats)',
                       args: { tool: 'string', repeats: ['boolean', 'keep approving until revoked'] }, required: ['tool'],
                       kind: 'policy', mutating: true, handler: async ({ tool: name, repeats = false }) => this.grant(name, repeats) }));
    reg.add(new Tool({ name: 'confirm.revoke', desc: 'withdraw approval for a tool', args: { tool: 'string' },
                       required: ['tool'], kind: 'policy', mutating: true,
                       handler: async ({ tool: name }) => ({ revoked: this.policy.revoke(name) }) }));
    reg.add(new Tool({ name: 'context.compact', desc: 'fold old turns into one summary, with a model if one is allowed',
                       args: { model: ['string', 'blank = extractive summary'] }, kind: 'context', mutating: true,
                       handler: async ({ model: name = '' } = {}) => this.compact(name) }));
    reg.add(new Tool({ name: 'agent.status', desc: 'the live picture: loop, budgets, window, plugins, jobs, tasks',
                       kind: 'loop', handler: async () => this.panel() }));
    reg.add(new Tool({ name: 'agent.step', desc: 'run exactly one observe-act cycle and return it',
                       args: { goal: ['string', 'set or replace the goal'] }, kind: 'loop', mutating: true,
                       handler: async ({ goal = '' } = {}) => this.step(goal) }));
    reg.add(new Tool({ name: 'agent.run', desc: 'hand a goal to the loop and let it drive (denied by default)',
                       args: { goal: 'string', max_steps: 'integer' }, required: ['goal'], kind: 'loop', mutating: true,
                       handler: async ({ goal, max_steps: maxSteps = 0 }) => (await this.run(goal, { max_steps: maxSteps })).brief() }));
  }

  cleanPolicyChanges(changes) {
    const out = {};
    for (const [key, value] of Object.entries(changes)) if (value !== undefined && value !== null) out[key] = value;
    return out;
  }

  // -- the funnel --------------------------------------------------------
  async call(name, args = null, source = 'loop') {
    const started = this.clock();
    const spec = this.registry.tools[name];
    const kwargs = { ...(this.tool_defaults[name] || {}), ...(args || {}) };
    if (!spec) {
      const record = new Call({ tool: name, args: kwargs, source,
                                error: `no tool '${name}'. ${this.registry.length} are registered; tools.list shows them`,
                                ms: Math.round((this.clock() - started) * 1000 * 100) / 100 });
      this.journal.push(record);
      await this.hooks.emit('tool_error', { tool: name, args: kwargs, error: record.error, source });
      return record;
    }
    const payload = { tool: name, args: { ...kwargs }, source, kind: spec.kind };
    const verdict = await this.hooks.emit('pre_tool', payload);
    let record;
    if (verdict.block) {
      record = new Call({ tool: name, kind: spec.kind, plugin: spec.plugin, args: verdict.args, blocked: true,
                          reason: verdict.reason, source });
    } else {
      const allowedArgs = verdict.args;
      const [ok, reason] = this.policy.check(spec, allowedArgs, source);
      if (!ok) {
        record = new Call({ tool: name, kind: spec.kind, plugin: spec.plugin, args: allowedArgs, blocked: true,
                            reason, source });
      } else {
        try {
          record = new Call({ tool: name, kind: spec.kind, plugin: spec.plugin, args: allowedArgs, ok: true,
                              value: await spec.call(allowedArgs), source });
        } catch (error) {
          record = new Call({ tool: name, kind: spec.kind, plugin: spec.plugin, args: allowedArgs,
                              error: error instanceof ToolError ? error.message : `${error.name}: ${error.message}`,
                              source });
        }
      }
    }
    record.ms = Math.round((this.clock() - started) * 1000 * 100) / 100;
    if (record.blocked) this.recordBlock(name, record.reason);
    this.journal.push(record);
    if (record.error) {
      await this.hooks.emit('tool_error', { tool: name, args: record.args, error: record.error, source });
    } else if (record.ok) {
      const post = await this.hooks.emit('post_tool', { tool: name, args: record.args, result: record.value, source });
      if (post.block) {
        record.blocked = true;
        record.reason = `post_tool veto: ${post.reason}`;
        record.ok = false;
      }
    }
    if (this.verbose) {
      console.log(`  ${name} ${record.blocked ? 'BLOCK' : record.error ? 'ERR' : 'ok'} ${record.ms}ms `
        + `${String(record.value || record.reason || '').slice(0, 90)}`);
    }
    return record;
  }

  recordBlock(name, reason) {
    /* Hook blocks land here; policy blocks already recorded themselves. */
    if (!this.policy.blocks.slice(-3).some((row) => row.tool === name && row.reason === reason)) {
      this.policy.blocks.push({ tool: name, reason, at: this.clock() });
    }
  }

  toolPrompt() {
    const lines = [];
    for (const spec of Object.values(this.registry.tools)) {
      if (matchesAny(spec.name, this.policy.deny)) continue;
      const flags = spec.danger ? '!' : spec.mutating ? '+' : spec.network ? '~' : ' ';
      const args = Object.keys(spec.args).map((key) => (spec.required.includes(key) ? `<${key}>` : `[${key}]`)).join(' ');
      lines.push(`${flags} ${spec.name} ${args}: ${spec.desc}`);
    }
    return lines.sort().join(NL);
  }

  visibleTools() {
    return this.registry.names().filter((name) => !matchesAny(name, this.policy.deny));
  }

  async think(step, goal) {
    const messages = this.context.asMessages();
    const call = await this.call(this.model, { messages, tools: this.visibleTools(), goal, step }, 'model');
    if (call.blocked || !call.ok) return [{ thought: '', final: '' }, call];
    return [call.value && typeof call.value === 'object' ? call.value : {}, call];
  }

  async fileSideChannels(action) {
    /* A model can hand back memory/task updates alongside its action. */
    let notes = 0, tasks = 0;
    const memory = action.memory;
    if (memory && typeof memory === 'object' && memory.key) {
      this.memory.write({ key: memory.key, value: memory.value === undefined ? '' : memory.value,
                          tags: memory.tags || '', kind: memory.kind || 'note' });
      notes += 1;
    }
    const task = action.task;
    if (task && typeof task === 'object' && task.subject) {
      this.tasks.create({ subject: task.subject, description: task.description || '', status: task.status || 'pending' });
      tasks += 1;
    }
    return { memory: notes, tasks };
  }

  // -- the loop ----------------------------------------------------------
  async run(goal, options = {}) {
    const { max_steps: maxSteps = 0, reset = true, script = null } = options;
    if (script) this.models.script({ script: 'default', turns: script });
    const run = new Run(goal, this.model, { started: this.clock() });
    this.policy.startRun();
    if (reset) this.context.reset(this.prompts.render(this));
    this.context.push('user', goal, 'goal');
    const steps = Math.trunc(maxSteps || this.policy.max_steps);
    await this.hooks.emit('plan', { tool: 'goal', args: { goal, steps } });
    let modelFailures = 0;
    for (let index = 0; index < steps; index += 1) {
      const [action, modelCall] = await this.think(index, goal);
      if (modelCall.blocked || !modelCall.ok) {
        modelFailures += 1;
        this.context.push('tool', modelCall.observation(), this.model);
        if (modelFailures >= 2) {
          run.status = 'model-failed';
          run.error = modelCall.reason || modelCall.error;
          break;
        }
        continue;
      }
      modelFailures = 0;
      const thought = String(action.thought || '');
      const filed = await this.fileSideChannels(action);
      run.memory_writes += filed.memory;
      run.tasks_created += filed.tasks;
      const turn = new Turn(index, { thought, action });
      if (action.final) {
        turn.final = String(action.final);
        this.context.push('assistant', turn.final, 'final');
        run.turns.push(turn);
        run.status = 'done';
        break;
      }
      const spec = action.call || {};
      const name = String(spec.tool || '');
      if (!name) {
        this.context.push('tool', { error: `turn ${index} had neither call nor final: ${JSON.stringify(action).slice(0, 200)}` }, 'loop');
        run.turns.push(turn);
        continue;
      }
      const record = await this.call(name, spec.args || {}, 'loop');
      turn.call = record;
      run.turns.push(turn);
      this.context.push('assistant', thought || name, 'thought');
      this.context.push('tool', record.observation(), name);
      if (this.context.needsCompaction()) {
        const outcome = await this.compact();
        if (outcome.compacted) run.compactions += 1;
      }
      if (this.scheduler.due().length) await this.tick();
    }
    if (run.status === 'running') run.status = 'steps-exhausted';
    run.seconds = Math.round((this.clock() - run.started) * 1000) / 1000;
    await this.hooks.emit('stop', { tool: run.status, args: { goal, turns: run.turns.length } });
    this.runs.push(run);
    return run;
  }

  /* One cycle for external control: a scheduler or a human can drive the loop. */
  async step(goal = '') {
    if (goal) this.context.push('user', goal, 'goal');
    const active = [...this.runs].reverse().find((run) => run.status === 'running') || null;
    const target = goal || (active ? active.goal
      : this.context.messages.length ? this.context.messages[0].content : '');
    const index = active ? active.turns.length : 0;
    const [action, modelCall] = await this.think(index, target);
    if (modelCall.blocked || !modelCall.ok) {
      return { ok: false, reason: modelCall.reason || modelCall.error, model: this.model };
    }
    const filed = await this.fileSideChannels(action);
    if (action.final) {
      this.context.push('assistant', action.final, 'final');
      return { ok: true, thought: action.thought || '', final: action.final, filed };
    }
    const spec = action.call || {};
    const record = await this.call(String(spec.tool || ''), spec.args || {}, 'step');
    this.context.push('assistant', action.thought || '', 'thought');
    this.context.push('tool', record.observation(), record.tool);
    return { ok: record.ok, thought: action.thought || '', call: record.brief(), filed };
  }

  async compact(modelName = '') {
    if (!modelName || !this.registry.has(modelName)) return this.context.compact();
    const digest = this.context.messages.map((m) => `${m.role}: ${m.content.slice(0, 300)}`).join(NL);
    const ask = await this.call(modelName, { messages: [
      { role: 'system', content: 'Summarise the transcript in <=3 sentences, keeping names, paths, numbers and any blocked call.' },
      { role: 'user', content: digest }], }, 'compact');
    if (ask.ok && ask.value && typeof ask.value === 'object') {
      const text = String(ask.value.final || ask.value.thought || '');
      return this.context.compact(text || null);
    }
    return this.context.compact();
  }

  async dispatch(name, args = null) { return (await this.call(name, args || {}, 'schedule')).observation(); }

  async tick(now = null) { return this.scheduler.tick((name, args) => this.dispatch(name, args), now); }

  hook(event, pattern = '*', call = null, toolName = '', name = '') {
    return this.hooks.add(event, pattern, call, toolName, name);
  }

  script(turns, name = 'default', append = false) { return this.models.script({ script: name, turns, append }); }

  grant(name, repeats = false) {
    /* Operator approval for a confirm-required tool; it survives runs until revoked. */
    this.registry.get(name);
    this.policy.grant(name, repeats);
    return { granted: name, repeats };
  }

  panel() {
    const kinds = {};
    for (const spec of Object.values(this.registry.tools)) (kinds[spec.kind] = kinds[spec.kind] || []).push(spec.name);
    const counts = Object.fromEntries(Object.entries(kinds).map(([key, list]) => [key, list.length]));
    const sorted = Object.fromEntries(Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)));
    return { agent: this.name, root: this.root, model: this.model, tools: this.registry.length, kinds: sorted,
             plugins: this.plugins.loaded, policy: this.policy.summary(), context: this.context.stats(),
             memory: this.memory.entries.length,
             skills: { installed: this.skills.list().map((pack) => pack.name), active: this.skills.active },
             tasks: this.tasks.summary(), jobs: this.scheduler.jobs.length, hooks: this.hooks.describe(),
             hook_firings: this.hooks.fired.length, blocks: this.policy.blocks.slice(-5),
             journal: this.journal.length,
             runs: this.runs.map((run) => ({ goal: run.goal.slice(0, 60), status: run.status, turns: run.turns.length })) };
  }
}

// ---------------------------------------------------------------------------
// examples — every subsystem, run for real
// ---------------------------------------------------------------------------

const scratch = (options = {}) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-example-'));
  return new Agent({ name: 'example', root, memory_path: path.join(root, '.agent', 'memory.json'),
                     skills_dir: null, ...options });
};

const examples = {
  'tool-loop': {
    description: 'A scripted model drives the funnel: write a file, read it back, answer.',
    run: async () => {
      const agent = scratch();
      agent.grant('fs.write');
      agent.script([
        { thought: 'put the note down', call: { tool: 'fs.write', args: { path: 'notes.md', content: 'alpha beta gamma\n' } } },
        { thought: 'read it back', call: { tool: 'fs.read', args: { path: 'notes.md' } } },
        { thought: 'answer', final: 'the note holds one line of three words' },
      ]);
      const run = await agent.run('write a note and prove you read it');
      return [`status=${run.status}  trail=${run.trail}`, `answer=${run.answer}`,
              `journal=${JSON.stringify(agent.journal.map((call) => call.tool))}`];
    },
  },
  'guardrails': {
    description: 'Deny-first patterns, a path sandbox, approvals and a vetoing hook.',
    run: async () => {
      const agent = scratch();
      fs.writeFileSync(path.join(agent.root, 'secret.txt'), 'do not read');
      agent.hook('pre_tool', 'fs.read', (payload) => (String((payload.args || {}).path || '').includes('secret')
        ? Promise.resolve({ block: true, reason: 'private file' }) : Promise.resolve(null)), '', 'no-secrets');
      agent.script([
        { thought: 'write without asking', call: { tool: 'fs.write', args: { path: 'a.txt', content: 'x' } } },
        { thought: 'read the secret', call: { tool: 'fs.read', args: { path: 'secret.txt' } } },
        { thought: 'escape the sandbox', call: { tool: 'fs.read', args: { path: '../outside.txt' } } },
        { thought: 'delete it', call: { tool: 'fs.delete', args: { path: 'a.txt' } } },
        { thought: 'give up', final: 'four different guards said no' },
      ]);
      const run = await agent.run('try to break the rules');
      return [...run.blocked.map((block) => `turn ${block.i} ${block.tool}: ${block.reason}`),
              `policy recorded ${agent.policy.blocks.length} blocks, spent ${agent.policy.calls} calls`];
    },
  },
  'memory-context': {
    description: 'The model files its own notes; a long window folds into a summary.',
    run: async () => {
      const agent = scratch({ budget_tokens: 900, keep_last: 3 });
      agent.grant('fs.write');
      agent.script([
        { thought: 'write a long file', memory: { key: 'notes', value: '3 lines of prose', kind: 'fact' },
          call: { tool: 'fs.write', args: { path: 'notes.md', content: `${'alpha beta gamma delta '.repeat(12)}\n` } } },
        { thought: 'read it back', call: { tool: 'fs.read', args: { path: 'notes.md' } } },
        { thought: 'recall', call: { tool: 'memory.search', args: { query: 'notes' } } },
        { thought: 'answer', final: '3 lines, and the window folded itself' },
      ]);
      const run = await agent.run('keep a note about the notes');
      return [`status=${run.status}  compactions during run=${run.compactions}  memory_writes=${run.memory_writes}`,
              `window=${JSON.stringify(agent.context.stats())}`, `memory=${agent.memory.notes()}`];
    },
  },
  'tasks-scheduler': {
    description: 'A job is a tool call through the same funnel; the loop ticks whatever is due.',
    run: async () => {
      const agent = scratch();
      const first = await agent.call('task.create', { subject: 'read the note' });
      await agent.call('task.create', { subject: 'summarise it', blocks: String(first.value.id) });
      await agent.call('schedule.add', { name: 'heartbeat', tool: 'echo', args: { text: 'beat' }, at: 1 });
      agent.script([
        { thought: 'finish the first task', call: { tool: 'task.update', args: { id: first.value.id, status: 'completed' } } },
        { thought: 'what is next', call: { tool: 'task.ready' } },
        { thought: 'answer', final: 'the queue moved on' },
      ]);
      const run = await agent.run('work the queue');
      return [`fired=${JSON.stringify(agent.scheduler.fired.map((row) => row.job))} -> `
              + `${JSON.stringify(agent.scheduler.fired[0].result)}`,
              `tasks=${JSON.stringify(agent.tasks.summary())}`, `status=${run.status}`];
    },
  },
  'skills-prompts': {
    description: 'A markdown skill pack joins the system prompt while it is active.',
    run: async () => {
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-skills-'));
      fs.writeFileSync(path.join(directory, 'notekeeper.md'), '---\nname: notekeeper\n'
        + 'description: keep short cited notes\nwhen_to_use: the user asks for notes\n'
        + 'tools: fs.write, nope.tool\n---\nAlways cite the file you read.\n');
      const agent = scratch({ skills_dir: [directory] });
      const used = await agent.call('skills.use', { name: 'notekeeper' });
      const rendered = await agent.call('prompt.get', { rendered: true });
      return [`installed=${JSON.stringify(agent.skills.list().map((pack) => pack.name))}`,
              `requires=${JSON.stringify(used.value.requires)}  missing=${JSON.stringify(used.value.missing_tools)}`,
              `prompt tokens=${rendered.value.tokens}  cites the skill=${rendered.value.rendered.includes('cite the file')}`];
    },
  },
  'plugin-pack': {
    description: 'Drop a .js into ./tools and it becomes a named, guardrailed, callable tool.',
    run: async () => {
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-tools-'));
      fs.writeFileSync(path.join(directory, 'wordcount.js'), "'use strict';\n"
        + "const { tool } = require('" + path.join(__dirname, 'agent.js').split('\\').join('\\\\') + "');\n"
        + 'module.exports = tool(\'words.count\', \'count the words in the text\',\n'
        + "  { args: { text: 'string' }, required: ['text'] },\n"
        + '  ({ text }) => ({ words: text.split(/\\s+/).filter(Boolean).length }));\n');
      const agent = scratch({ tools_dir: directory });
      const called = await agent.call('words.count', { text: 'one two three' });
      await agent.call('policy.set', { deny: ['words.*'] });
      const blocked = await agent.call('words.count', { text: 'one two three' });
      return [`plugins=${JSON.stringify(agent.plugins.loaded)}  call ok=${called.ok} value=${JSON.stringify(called.value)}`,
              `blocked=${blocked.reason}`];
    },
  },
  'nested-agents': {
    description: 'A worker agent is just another tool in the planner’s registry.',
    run: async () => {
      const worker = scratch({ name: 'worker' });
      worker.script([{ call: { tool: 'echo', args: { text: 'counted 3 rows' } } }, { final: '3 rows counted' }]);
      const planner = scratch({ name: 'planner' });
      planner.registry.add(new Tool({ name: 'worker.ask', desc: 'hand a goal to the worker agent',
                                      args: { goal: 'string' }, required: ['goal'], kind: 'tool', mutating: true,
                                      handler: async ({ goal }) => (await worker.run(goal)).answer }));
      planner.script([{ call: { tool: 'worker.ask', args: { goal: 'count the rows' } } },
                      { final: 'the worker says 3 rows counted' }]);
      const run = await planner.run('delegate the counting');
      return [`planner trail=${run.trail}`, `worker journal=${JSON.stringify(worker.journal.map((call) => call.tool))}`,
              `answer=${run.answer}`];
    },
  },
  'local-endpoints': {
    description: 'ollama, an OpenAI-compatible server and llama.cpp are ordinary tools. Set AGENT_LIVE=1 to answer.',
    run: async () => {
      const agent = scratch();
      const lines = ['model.ollama', 'model.openai', 'model.llamacpp'].map((name) => {
        const spec = agent.registry.get(name);
        const endpoint = Object.keys(spec.args).filter((key) => !['messages', 'tools', 'goal', 'step'].includes(key));
        return `${name}: network=${spec.network} endpoint args=${JSON.stringify(endpoint.sort())}`;
      });
      const live = process.env.AGENT_LIVE === '1';
      const probe = scratch({ model: 'model.ollama', allow_network: live });
      const answer = await probe.call('model.ollama',
        { messages: [{ role: 'user', content: 'reply with {"final": "ping"}' }], tools: ['echo'] });
      lines.push(`live=${live} ok=${answer.ok} detail=${JSON.stringify(answer.ok ? answer.value : (answer.reason || answer.error)).slice(0, 220)}`);
      return lines;
    },
  },
  'llm-call': {
    description: 'One plain request to one model endpoint - no loop, no tools, just a reply and a log line.',
    run: async () => {
      const agent = scratch();
      agent.script([{ thought: '12 * 4 is 48', final: '48' }]);
      const reply = await agent.call('model.scripted', {
        messages: [{ role: 'system', content: 'reply with one JSON action' },
                   { role: 'user', content: 'what is 12 * 4?' }],
        tools: ['echo'], goal: 'answer the arithmetic', step: 0 });
      const probes = [];
      for (const name of ['model.scripted', 'model.openai', 'model.llamacpp']) {
        probes.push((await agent.call('policy.check', { tool: name, args: {} })).value);
      }
      const log = await agent.call('model.log', { limit: 2 });
      return [`action back: ${JSON.stringify(reply.value)}`,
              `model.log: ${JSON.stringify(log.value)}`,
              probes.map((p) => `${p.tool} allowed=${p.allowed}${p.allowed ? '' : ' - ' + p.reason}`).join(' | '),
              'agent.run() is this same call on a loop: think, act, observe, until \'final\' or the step budget'];
    },
  },
  'rag': {
    description: 'Index chunks in memory, retrieve one, answer only with what came back.',
    run: async () => {
      const corpus = { 'chunk:funnel': 'The harness funnel is registry -> hooks -> policy -> handler.',
                       'chunk:budgets': 'Budgets cap writes, deletes, tool calls and wall-clock seconds.',
                       'chunk:models': 'Model endpoints are ordinary network tools, off by default.' };
      const agent = scratch();
      for (const [key, value] of Object.entries(corpus)) {
        await agent.call('memory.write', { key, value, kind: 'chunk', tags: 'harness' });
      }
      const found = await agent.call('memory.search', { query: 'what caps writes', kind: 'chunk' });
      const hit = found.value[0];
      agent.script([
        { thought: 'retrieve first', call: { tool: 'memory.search', args: { query: 'what caps writes', kind: 'chunk' } } },
        { thought: 'put the retrieved chunk in the window',
          call: { tool: 'context.push', args: { role: 'user', content: hit.value, name: 'rag' } } },
        { thought: 'answer from the chunk', final: `${hit.key}: ${hit.value}` },
      ]);
      const run = await agent.run('what caps writes? answer only from the indexed chunks');
      return [`indexed ${Object.keys(corpus).length} chunks, top hit=${hit.key}`, `trail=${run.trail}`,
              `status=${run.status}`, `answer=${run.answer}`, `window=${JSON.stringify(agent.context.stats())}`];
    },
  },
  'agentic-rag': {
    description: 'The agent chooses its own retrieval queries and searches again when a hit comes back empty.',
    run: async () => {
      const agent = scratch();
      const corpus = { 'chunk:scheduler': 'A job is a tool call on a timer; the loop ticks whatever is due.',
                       'chunk:tasks': 'Tasks block other tasks until they are marked completed.' };
      for (const [key, value] of Object.entries(corpus)) {
        await agent.call('memory.write', { key, value, kind: 'chunk', tags: 'harness' });
      }
      agent.script([
        { thought: 'start with the words the goal used',
          call: { tool: 'memory.search', args: { query: 'vector embedding cosine', kind: 'chunk' } } },
        { thought: 'that came back empty; ask differently',
          call: { tool: 'memory.search', args: { query: 'timer every due', kind: 'chunk' } } },
        { thought: 'the second hit answers it', final: 'a job is a tool call on a timer' },
      ]);
      const run = await agent.run('how does the scheduler work?');
      const searches = agent.journal.filter((call) => call.tool === 'memory.search');
      const keys = (call) => JSON.stringify((call.value || []).map((entry) => entry.key));
      return [`trail=${run.trail}`, `query 1 -> ${keys(searches[0])}`, `query 2 -> ${keys(searches[1])}`,
              `status=${run.status} answer=${run.answer}`];
    },
  },
  'structured-output': {
    description: 'A post_tool hook is the schema: a malformed record is vetoed, so the model refiles it as JSON.',
    run: async () => {
      const agent = scratch();
      const validate = (payload) => {
        try { JSON.parse(String((payload.args || {}).value)); return null; }
        catch { return { block: true, reason: 'structured-output: value must be one JSON object' }; }
      };
      agent.hook('post_tool', 'memory.write', validate, '', 'schema');
      agent.script([
        { thought: 'file the record',
          call: { tool: 'memory.write', args: { key: 'invoice', value: '{ total: 42, currency: EUR }' } } },
        { thought: 'keys quoted and refiled',
          call: { tool: 'memory.write', args: { key: 'invoice', value: JSON.stringify({ total: 42, currency: 'EUR' }) } } },
        { thought: 'done', final: 'the second write passed the schema hook' },
      ]);
      const run = await agent.run('file the invoice as structured data');
      const record = await agent.call('memory.read', { key: 'invoice' });
      return [`trail=${run.trail}`, `vetoed=${JSON.stringify(run.blocked.map((block) => block.reason))}`,
              `record now=${record.value.value}`, `status=${run.status}`];
    },
  },
  'thinking': {
    description: 'No reasoning, no answer: a hook vetoes model actions that arrive without a thought.',
    run: async () => {
      const agent = scratch();
      agent.hook('post_tool', 'model.scripted',
        (payload) => (String(((payload.result || {}).thought) || '') ? null
          : { block: true, reason: 'thinking: show the reasoning first' }), '', 'show-your-work');
      agent.script([
        { call: { tool: 'echo', args: { text: '42, guessed' } } },
        { thought: '41 + 1: carry the one, so 42', call: { tool: 'echo', args: { text: '42' } } },
        { thought: 'answered with the reasoning shown', final: '42' },
      ]);
      const run = await agent.run('answer with reasoning');
      const vetoed = agent.journal.filter((call) => call.tool === 'model.scripted' && call.blocked);
      return [`model calls vetoes=${vetoed.length} (${vetoed.length ? vetoed[0].reason : 'none'})`,
              `turns that survived=${JSON.stringify(run.turns.map((turn) => [turn.i, (turn.thought || '').slice(0, 26)]))}`,
              `trail=${run.trail}`, `status=${run.status} answer=${run.answer}`];
    },
  },
  'judge': {
    description: 'A judge tool scores the draft; the loop rewrites until the bar is cleared.',
    run: async () => {
      const agent = scratch();
      const judge = ({ text }) => {
        const words = String(text).split(/\s+/).filter(Boolean).length;
        const score = Math.min(5, Math.floor(words / 4));
        return { score, verdict: score >= 3 ? 'accept' : 'revise', why: `${words} words, ${score}/5` };
      };
      agent.registry.add(new Tool({ name: 'judge.score', desc: 'score a draft 0-5; 3 is the bar',
                                    args: { text: 'string' }, required: ['text'], kind: 'tool', handler: judge }));
      const draft = 'the harness calls tools';
      const better = 'one funnel, repeated: registry, pre_tool hooks, policy, handler, post_tool hooks, '
        + 'and the context window sees the observation';
      agent.script([
        { thought: 'score the first draft', call: { tool: 'judge.score', args: { text: draft } } },
        { thought: 'too thin; write the rewrite down',
          call: { tool: 'memory.write', args: { key: 'draft', value: better } } },
        { thought: 'score the rewrite', call: { tool: 'judge.score', args: { text: better } } },
        { thought: 'accepted', final: 'the second draft cleared the judge' },
      ]);
      const run = await agent.run('write something the judge accepts');
      const scores = agent.journal.filter((call) => call.tool === 'judge.score' && call.ok)
        .map((call) => call.value.score);
      return [`trail=${run.trail}`, `scores=${JSON.stringify(scores)} (bar is 3)`,
              `status=${run.status} answer=${run.answer}`,
              `journal=${JSON.stringify(agent.journal.map((call) => call.tool))}`];
    },
  },
  'map-reduce': {
    description: 'Split a file into blocks, map a tool over each one, reduce the results to a single answer.',
    run: async () => {
      const agent = scratch();
      fs.writeFileSync(path.join(agent.root, 'rows.csv'), 'a,b,c\n1,2,3\n4,5,6\n7,8,9\n');
      agent.registry.add(new Tool({ name: 'map.sum', desc: 'add up one block of csv rows',
                                    args: { block: 'string' }, required: ['block'], kind: 'tool',
                                    handler: ({ block }) => {
                                      const rows = String(block).trim().split('\n').filter(Boolean);
                                      return { rows: rows.length,
                                               sum: rows.reduce((total, row) => total
                                                 + row.split(',').reduce((n, cell) => n + Number(cell), 0), 0) };
                                    } }));
      agent.registry.add(new Tool({ name: 'reduce.combine', desc: 'fold the mapped results into one',
                                    args: { results: 'array' }, required: ['results'], kind: 'tool',
                                    handler: ({ results }) => ({ blocks: results.length,
                                                                 rows: results.reduce((n, r) => n + r.rows, 0),
                                                                 total: results.reduce((n, r) => n + r.sum, 0) }) }));
      agent.script([
        { thought: 'read the file', call: { tool: 'fs.read', args: { path: 'rows.csv' } } },
        { thought: 'map block 1', call: { tool: 'map.sum', args: { block: '1,2,3' } } },
        { thought: 'map block 2', call: { tool: 'map.sum', args: { block: '4,5,6' } } },
        { thought: 'map block 3', call: { tool: 'map.sum', args: { block: '7,8,9' } } },
        { thought: 'reduce the three maps',
          call: { tool: 'reduce.combine', args: { results: [{ rows: 1, sum: 6 }, { rows: 1, sum: 15 }, { rows: 1, sum: 24 }] } } },
        { thought: 'answered', final: '3 blocks, 3 rows, total 45' },
      ]);
      const run = await agent.run('add up rows.csv block by block');
      const mapped = agent.journal.filter((call) => call.tool === 'map.sum' && call.ok).map((call) => call.value);
      const reduced = (agent.journal.find((call) => call.tool === 'reduce.combine' && call.ok) || {}).value || {};
      return [`trail=${run.trail}`, `map -> ${JSON.stringify(mapped)}`, `reduce -> ${JSON.stringify(reduced)}`,
              `status=${run.status} answer=${run.answer}`];
    },
  },
  'chat-guardrail': {
    description: 'A hook in front of the model: on-topic goals get a system note, off-topic ones never reach it.',
    run: async () => {
      const agent = scratch();
      const guard = (payload) => {
        const goal = String((payload.args || {}).goal || '');
        if (['password', 'crypto'].some((word) => goal.toLowerCase().includes(word))) {
          return { block: true, reason: `chat guardrail: ${goal.slice(0, 40)} is out of scope for this assistant` };
        }
        return { args: { messages: [{ role: 'system', content: 'Answer only about this codebase.' }]
                           .concat((payload.args || {}).messages || []) } };
      };
      agent.hook('pre_tool', 'model.*', guard, '', 'guardrail');
      agent.script([{ thought: 'off topic', final: 'here is a password trick' }]);
      const off = await agent.run('how do I reset someone password');
      agent.script([{ thought: 'on topic', final: 'one funnel: registry, hooks, policy, handler' }]);
      const on = await agent.run('what is the funnel?');
      const asked = agent.journal.filter((call) => call.tool === 'model.scripted' && !call.blocked);
      const note = asked.length ? ((asked[asked.length - 1].args.messages || [{}])[0].content || '') : '';
      return [`off-topic: status=${off.status} error=${off.error}`,
              `the model was never asked for it (model.log holds ${agent.models.log.length} calls)`,
              `on-topic: status=${on.status} answer=${on.answer}`,
              `first message the model did see: ${JSON.stringify(note.slice(0, 70))}`];
    },
  },
  'web-page': {
    description: 'browser.open cuts a page down to title, text and links - offline, over a data: URL.',
    run: async () => {
      const page = '<html><head><title>harness notes</title><style>body{color:red}</style></head>'
        + '<body><script>tracking()</script><p>one funnel: registry, hooks, policy, handler</p>'
        + '<a href="/hooks">hooks</a><a href="/policy">policy</a></body></html>';
      const url = `data:text/html;charset=utf-8,${encodeURIComponent(page)}`;
      const agent = scratch();
      const probe = (await agent.call('policy.check', { tool: 'browser.open', args: { url } })).value;
      await agent.call('policy.set', { allow_network: true });
      const seen = (await agent.call('browser.open', { url })).value;
      agent.script([
        { thought: 'open the page', call: { tool: 'browser.open', args: { url } } },
        { thought: 'answer from the reduced text', final: `${seen.title}, ${seen.links.length} links` },
      ]);
      const run = await agent.run('what does the page say?');
      return [`guardrail first: allowed=${probe.allowed} (${probe.reason})`,
              `network on -> title=${JSON.stringify(seen.title)}`,
              `text=${JSON.stringify(seen.text.slice(0, 58))}`,
              `links=${JSON.stringify(seen.links)}`, `trail=${run.trail} status=${run.status} answer=${run.answer}`];
    },
  },
  'coding-agent': {
    description: 'Read the spec, write the module, run the tests, patch the failure, run them again.',
    run: async () => {
      const agent = scratch();
      fs.writeFileSync(path.join(agent.root, 'spec.md'), '# spec\nadd(a, b) returns the sum of two integers.\n');
      fs.writeFileSync(path.join(agent.root, 'test_app.js'),
        'const { add } = require(\'./app.js\');\n'
        + 'if (add(2, 3) !== 5) { console.error(\'AssertionError: add must sum\'); process.exit(1); }\n'
        + 'console.log(\'tests passed\');\n');
      // what a coding agent needs opened, and nothing more: the shell is allowed and approved, deletes still denied
      agent.policy.set({ deny: ['agent.run', 'fs.delete'] });
      agent.grant('fs.write', true);
      agent.grant('fs.edit', true);
      agent.grant('shell.run', true);
      const runCmd = `${JSON.stringify(process.execPath)} test_app.js`;
      agent.script([
        { thought: 'read the spec', call: { tool: 'fs.read', args: { path: 'spec.md' } } },
        { thought: 'write the module, with a slip in it',
          call: { tool: 'fs.write', args: { path: 'app.js', content: 'module.exports.add = (a, b) => a - b;\n' } } },
        { thought: 'run the tests', call: { tool: 'shell.run', args: { cmd: runCmd } } },
        { thought: 'the failure says sum, not difference; patch that one line',
          call: { tool: 'fs.edit', args: { path: 'app.js', find: 'a - b', replace: 'a + b' } } },
        { thought: 'run them again', call: { tool: 'shell.run', args: { cmd: runCmd } } },
        { thought: 'green', final: 'tests pass: app.js now sums' },
      ]);
      const run = await agent.run('implement spec.md and make the tests pass');
      const tests = agent.journal.filter((call) => call.tool === 'shell.run' && call.ok).map((call) => call.value);
      const lastLine = (row) => ((row.stdout || row.stderr).trim().split('\n').filter(Boolean).pop()) || '';
      const patched = (agent.journal.find((call) => call.tool === 'fs.edit' && call.ok) || {}).value || {};
      return [`guards for this job: deny=${JSON.stringify(agent.policy.deny)} `
              + `granted=${JSON.stringify(Object.keys(agent.policy.granted).sort())}`,
              `trail=${run.trail}`,
              `test runs=${JSON.stringify(tests.map((row) => ({ code: row.code, 'last line': lastLine(row).slice(0, 44) })))}`,
              `patch=${JSON.stringify({ replacements: patched.replacements, matches: patched.matches })}`,
              `status=${run.status} answer=${run.answer}`];
    },
  },
};

async function runExamples(names = null) {
  const wanted = names && names.length ? names : Object.keys(examples);
  const unknown = wanted.filter((name) => !(name in examples));
  if (unknown.length) {
    console.log(`no example ${unknown.join(', ')}. available: ${Object.keys(examples).join(', ')}`);
    return 2;
  }
  for (const name of wanted) {
    console.log(`\n=== ${name} - ${examples[name].description}`);
    for (const line of await examples[name].run()) console.log(`  ${line}`);
  }
  return 0;
}

function printRun(run) {
  console.log(`\nstatus: ${run.status}   model: ${run.model}   ${run.seconds}s`);
  console.log(`trail : ${run.trail}`);
  for (const turn of run.turns) {
    const record = turn.call;
    const flag = record && record.blocked ? 'BLOCK' : record && record.error ? 'ERR  ' : 'ok   ';
    const detail = record ? (record.reason || record.error || JSON.stringify(record.value)) : '';
    const name = record ? record.tool : '(no call)';
    console.log(`  ${turn.i} ${flag} ${name.padEnd(16)} ${(turn.thought || '').slice(0, 38).padEnd(38)} ${String(detail).slice(0, 70)}`);
  }
  console.log(`answer: ${run.answer}`);
  if (run.blocked.length) {
    console.log('blocked:');
    for (const block of run.blocked) console.log(`  turn ${block.i} ${block.tool}: ${block.reason}`);
  }
}

function demoAgent(root) {
  const agent = new Agent({ name: 'demo', root,
    memory_path: path.join(root, '.agent', 'memory.json'), budget_tokens: 900 });
  agent.hook('pre_tool', 'fs.*', (payload) => (
    String((payload.args || {}).path || '').includes('secret')
      ? { block: true, reason: 'the no-secrets hook refuses private files' } : null), '', 'no-secrets');
  agent.grant('fs.write');
  agent.script([
    { thought: 'put the note down', call: { tool: 'fs.write', args: { path: 'notes.md', content: '# notes\nalpha beta gamma delta' } } },
    { thought: 'read it back', call: { tool: 'fs.read', args: { path: 'notes.md' } } },
    { thought: 'peek at something private', call: { tool: 'fs.read', args: { path: 'secret.txt' } } },
    { thought: 'try to delete it', call: { tool: 'fs.delete', args: { path: 'notes.md' } } },
    { thought: 'remember the count', memory: { key: 'note-length', value: '4 words', kind: 'fact' }, task: { subject: 'expand the notes' } },
    { thought: 'count the words', call: { tool: 'memory.search', args: { query: 'note' } } },
    { thought: 'answer', final: 'notes.md holds 4 words; secret read and delete were both blocked' },
  ]);
  return agent;
}

async function main(argv = []) {
  const command = argv.length ? argv[0] : 'demo';
  if (command === 'demo') {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-demo-'));
    fs.writeFileSync(path.join(root, 'secret.txt'), 'do not read', 'utf8');
    const agent = demoAgent(root);
    const panel = agent.panel();
    console.log(`workspace ${agent.root}`);
    console.log(`panel    ${JSON.stringify(panel.kinds)} tools across ${panel.tools} names; plugins=${panel.plugins.length ? panel.plugins.join(',') : 'none'}`);
    printRun(await agent.run('summarise the notes and prove the guardrails bite'));
    console.log(`\ntasks   ${JSON.stringify(agent.tasks.summary())}`);
    console.log(`memory  ${JSON.stringify(agent.memory.notes())}`);
    console.log(`blocks  ${JSON.stringify(agent.policy.blocks)}`);
    console.log(`context ${JSON.stringify(agent.context.stats())}`);
    console.log(`hooks   ${JSON.stringify(agent.hooks.fired.slice(-3)).slice(0, 400)}`);
    return 0;
  }
  if (command === 'tools') {
    const agent = new Agent({ root: process.cwd() });
    for (const spec of agent.registry.brief()) {
      const marks = [['!', spec.danger], ['+', spec.mutating], ['~', spec.network]]
        .filter((pair) => pair[1]).map((pair) => pair[0]).join('');
      console.log(`${spec.kind.padEnd(9)}${spec.name.padEnd(20)}${marks.padEnd(4)}${spec.desc.slice(0, 70)}`);
    }
    console.log(`\n${agent.registry.length} tools, kinds=${JSON.stringify(agent.panel().kinds)}`);
    return 0;
  }
  if (command === 'skills') {
    console.log(JSON.stringify(new Agent({ root: process.cwd() }).skills.list(), null, 2));
    return 0;
  }
  if (command === 'panel') {
    console.log(JSON.stringify(new Agent({ root: process.cwd() }).panel(), null, 2));
    return 0;
  }
  if (command === 'examples') return runExamples(argv.slice(1));
  if (command === 'run') {
    const goal = argv.length > 1 ? argv[1] : 'hello';
    const at = argv.indexOf('--model');
    const model = at >= 0 ? argv[at + 1] : 'model.scripted';
    const agent = new Agent({ root: process.cwd(), model, allow_network: model !== 'model.scripted' });
    if (model === 'model.scripted') {
      await agent.run(goal, { script: [{ thought: 'say hi', call: { tool: 'echo', args: { text: goal } } },
                                       { thought: 'done', final: `echoed "${goal}"` }] });
    }
    printRun(agent.runs[agent.runs.length - 1]);
    return 0;
  }
  console.log(USAGE);
  return 2;
}

if (require.main === module) {
  main(process.argv.slice(2)).then((code) => { process.exitCode = code; },
    (error) => { console.error(error && error.stack || String(error)); process.exitCode = 1; });
}

module.exports = {
  Agent, Tool, Registry, defaultRegistry, tool, adopt, discover, Policy, Hooks, Memory, Context,
  Prompts, Skills, Tasks, Scheduler, Models, Builtins, Call, Turn, Run, examples, runExamples, main,
  estimateTokens, resolvePath, under, coerce, firstJson, fnmatch, matchesAny, renderChatml, splitCommand,
  globToRegExp, ToolError, Blocked, UnknownTool, SandboxError,
  BASE_PROMPT, PERSONA_PROMPT, PATH_KEYS, HOOK_EVENTS, TASK_STATUSES, JSON_TYPES,
  DEFAULT_TOOLS_DIR, ENTRY_POINT_GROUP, MODEL_GROUP, USAGE,
};






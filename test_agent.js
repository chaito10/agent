/*
 * test_agent.js — the Node twin's own suite, plus parity with agent.py and agent.html.
 *
 *     node test_agent.js            run everything
 *     node test_agent.js funnel     run the groups whose name matches
 *
 * Zero dependencies: a tiny async runner over node:assert.
 */
'use strict';

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const H = require('./agent.js');
const { Agent, Tool, Registry, tool, Policy, Hooks, Memory, Context, Prompts, Skills, Tasks, Scheduler,
        Models, Call, Turn, Run, discover, firstJson, fnmatch, estimateTokens, resolvePath, under,
        ToolError, Blocked, UnknownTool, SandboxError, examples, runExamples } = H;

const workspace = () => fs.mkdtempSync(path.join(os.tmpdir(), 'agent-test-'));
const makeAgent = (options = {}) => new Agent({ root: workspace(), skills_dir: null, memory_path: null, ...options });

const groups = [];
const suite = (name, define) => { const cases = []; const test = (label, fn) => cases.push([label, fn]); define(test); groups.push([name, cases]); };

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

// ---------------------------------------------------------------------------
suite('funnel', (test) => {
  test('every component registers tools into one flat registry', async () => {
    const agent = makeAgent();
    const kinds = new Set(Object.values(agent.registry.tools).map((spec) => spec.kind));
    for (const kind of ['tool', 'skill', 'memory', 'context', 'hook', 'policy', 'prompt', 'task', 'schedule', 'model', 'loop']) {
      assert.ok(kinds.has(kind), `no tool of kind ${kind}`);
    }
    assert.strictEqual(agent.registry.length, 57);
  });

  test('unknown tools come back with a useful message', async () => {
    const agent = makeAgent();
    const record = await agent.call('nope.tool', {});
    assert.strictEqual(record.ok, false);
    assert.match(record.error, /no tool 'nope\.tool'/);
    assert.match(record.error, /tools\.list shows them/);
  });

  test('args are coerced to the declared type', async () => {
    const agent = makeAgent();
    const record = await agent.call('echo', { text: 42 });
    assert.strictEqual(record.value.echo, '42');
    assert.strictEqual((await agent.call('time.now', {})).ok, true);
  });

  test('unexpected and missing args are tool errors, not crashes', async () => {
    const agent = makeAgent();
    assert.match((await agent.call('echo', { text: 'a', extra: 1 })).error, /unexpected arg 'extra'/);
    assert.match((await agent.call('echo', {})).error, /missing required arg\(s\): text/);
  });

  test('the journal records every call in order, with timing', async () => {
    const agent = makeAgent();
    await agent.call('echo', { text: 'one' });
    await agent.call('time.now', {});
    assert.deepStrictEqual(agent.journal.map((row) => row.tool), ['echo', 'time.now']);
    assert.ok(agent.journal.at(-1).ms >= 0);
  });

  test('tools.call is the meta plug and goes through the same funnel', async () => {
    const agent = makeAgent();
    const outcome = await agent.call('tools.call', { tool: 'echo', args: { text: 'via meta' } });
    assert.deepStrictEqual(outcome.value, { tool: 'echo', result: { echo: 'via meta' } });
    const denied = await agent.call('tools.call', { tool: 'fs.delete', args: { path: 'x' } });
    assert.strictEqual(denied.value.blocked, true);
    assert.match(denied.value.reason, /denied by guardrail/);
  });

  test('operator defaults merge into every call', async () => {
    const agent = makeAgent();
    fs.writeFileSync(path.join(agent.root, 'notes.md'), 'a');
    assert.match((await agent.call('fs.read', {})).error, /missing required arg\(s\): path/);
    agent.tool_defaults['fs.read'] = { path: 'notes.md' };
    assert.match((await agent.call('fs.read', {})).value.path, /notes\.md$/);
    fs.writeFileSync(path.join(agent.root, 'other.txt'), 'b');
    assert.match((await agent.call('fs.read', { path: 'other.txt' })).value.path, /other\.txt$/);
    assert.strictEqual(agent.journal.at(-1).args.path, 'other.txt');   // merged args are what policy sees
  });

  test('a raising tool becomes an error record, never a throw', async () => {
    const agent = makeAgent();
    agent.registry.add(new Tool({ name: 'boom', desc: 'always fails', args: {}, kind: 'tool',
                                  handler: async () => { throw new ToolError('kaput'); } }));
    const record = await agent.call('boom', {});
    assert.strictEqual(record.error, 'kaput');
    assert.strictEqual(record.ok, false);
  });
});

// ---------------------------------------------------------------------------
suite('action contract', (test) => {
  test('a scripted model drives the loop to a final answer', async () => {
    const agent = makeAgent();
    agent.script([{ thought: 'say hi', call: { tool: 'echo', args: { text: 'hi' } } }, { final: 'done' }]);
    const run = await agent.run('say hi');
    assert.strictEqual(run.status, 'done');
    assert.strictEqual(run.trail, 'echo');
    assert.strictEqual(run.answer, 'done');
    assert.strictEqual(run.ok, true);
  });

  test('a turn with neither call nor final is reported back to the model', async () => {
    const agent = makeAgent();
    agent.script([{ thought: 'confused' }, { final: 'recovered' }]);
    const run = await agent.run('be confused');
    assert.strictEqual(run.status, 'done');
    const looped = agent.context.messages.filter((row) => row.name === 'loop');
    assert.match(looped.at(-1).content, /neither call nor final/);
  });

  test('side channels file memory and tasks without a tool call', async () => {
    const agent = makeAgent({ memory_path: path.join(workspace(), 'memory.json') });
    agent.script([{ thought: 'note it', memory: { key: 'k', value: 'v', kind: 'fact' }, task: { subject: 'later' } },
                  { final: 'filed' }]);
    const run = await agent.run('file things');
    assert.strictEqual(run.memory_writes, 1);
    assert.strictEqual(run.tasks_created, 1);
    assert.strictEqual(agent.memory.entries[0].kind, 'fact');
    assert.strictEqual(agent.tasks.items[0].subject, 'later');
  });

  test('run brief and observation shapes are stable', async () => {
    const agent = makeAgent();
    agent.script([{ call: { tool: 'echo', args: { text: 'x' } } }, { final: 'y' }]);
    const run = await agent.run('brief me');
    const brief = run.brief();
    assert.deepStrictEqual(Object.keys(brief).sort(),
      ['answer', 'blocked', 'goal', 'model', 'seconds', 'status', 'trail', 'turns'].sort());
    const record = await agent.call('fs.read', { path: 'missing.txt' });
    assert.deepStrictEqual(record.observation(), { error: record.error, tool: 'fs.read' });
    assert.strictEqual(new Call({ blocked: true, tool: 't', reason: 'r' }).observation().blocked, true);
  });
});

// ---------------------------------------------------------------------------
suite('policy', (test) => {
  test('deny wins over everything, and deny patterns are fnmatch', async () => {
    const agent = makeAgent();
    for (const name of ['agent.run', 'fs.delete', 'shell.run']) {
      const [ok, reason] = agent.policy.check(agent.registry.get(name), {});
      assert.strictEqual(ok, false);
      assert.match(reason, /denied by guardrail rule/);
    }
    assert.ok(fnmatch('shell.run', 'shell.*'));
    assert.ok(fnmatch('shell.run.x', 'shell.*'));      // fnmatch's * crosses dots, as in agent.py
    assert.ok(!fnmatch('fs.read', '*.fs'));            // the pattern is anchored at the end
  });

  test('the allow list, when set, hides everything else', async () => {
    const agent = makeAgent();
    agent.policy.set({ deny: [], allow: ['echo'] });
    const [ok, reason] = agent.policy.check(agent.registry.get('fs.read'), {});
    assert.strictEqual(ok, false);
    assert.match(reason, /allow list/);
  });

  test('the operator is never locked out of the guardrails', async () => {
    // a hand-typed allow list can forget policy.set itself; that must not strand the
    // panel, while every model-driven call to the same tool keeps obeying the rule
    const agent = makeAgent();
    agent.policy.set({ allow: ['policy.allow_network'] });
    const blocked = await agent.call('policy.set', { allow_network: true }, 'model');
    assert.ok(blocked.blocked);
    assert.match(blocked.reason, /allow list/);
    const freed = await agent.call('policy.set', { allow: [] }, 'panel');
    assert.ok(freed.ok, freed.reason);
    assert.deepStrictEqual(agent.policy.allow, []);
    // dry-run blocks every mutating tool, policy.set included
    await agent.call('policy.set', { dry_run: true }, 'panel');
    assert.ok((await agent.call('fs.write', { path: 'a.txt', text: 'x' })).blocked);
    assert.ok((await agent.call('policy.set', { dry_run: false }, 'panel')).ok);
  });

  test('network tools are refused until allow_network', async () => {
    const agent = makeAgent();
    let [ok, reason] = agent.policy.check(agent.registry.get('http.get'), { url: 'http://x' });
    assert.strictEqual(ok, false);
    assert.match(reason, /network tools are off/);
    agent.policy.set({ allow_network: true });
    [ok] = agent.policy.check(agent.registry.get('http.get'), { url: 'http://x' });
    assert.strictEqual(ok, true);
  });

  test('paths are sandboxed to root plus extra_dirs', async () => {
    const agent = makeAgent();
    const [ok, reason] = agent.policy.check(agent.registry.get('fs.read'), { path: '../../etc/passwd' });
    assert.strictEqual(ok, false);
    assert.match(reason, /escapes the sandbox/);
    const extra = workspace();
    fs.writeFileSync(path.join(extra, 'a.txt'), 'a');
    agent.policy.set({ extra_dirs: [extra] });
    assert.strictEqual(agent.policy.check(agent.registry.get('fs.read'), { path: path.join(extra, 'a.txt') })[0], true);
  });

  test('confirm-required tools need a one-shot approval unless repeats', async () => {
    const agent = makeAgent();
    assert.match((await agent.call('fs.write', { path: 'a.txt', content: 'x' })).reason, /needs approval/);
    agent.grant('fs.write');
    assert.strictEqual((await agent.call('fs.write', { path: 'a.txt', content: 'x' })).ok, true);
    assert.match((await agent.call('fs.write', { path: 'b.txt', content: 'x' })).reason, /needs approval/);
    agent.grant('fs.write', true);
    assert.strictEqual((await agent.call('fs.write', { path: 'c.txt', content: 'x' })).ok, true);
    assert.strictEqual((await agent.call('fs.write', { path: 'd.txt', content: 'x' })).ok, true);
    assert.strictEqual(agent.policy.revoke('fs.write'), true);
  });

  test('approvals survive a run, budgets reset per run', async () => {
    const agent = makeAgent();
    const turns = () => [{ call: { tool: 'fs.write', args: { path: 'a.txt', content: 'x' } } }, { final: 'ok' }];
    agent.grant('fs.write', true);
    agent.script(turns());
    await agent.run('first');
    assert.strictEqual(agent.policy.calls, 3);          // two model calls plus the write
    agent.script(turns());
    await agent.run('second');
    assert.strictEqual(agent.policy.calls, 3);          // the budget reset, the grant did not
    assert.strictEqual(agent.policy.writes, 1);
    assert.ok('fs.write' in agent.policy.granted);
  });

  test('call, write and delete budgets are separate', async () => {
    const agent = makeAgent();
    agent.grant('fs.write', true);
    agent.policy.set({ max_writes: 1, max_deletes: 0, max_tool_calls: 3, deny: [] });
    assert.strictEqual((await agent.call('fs.write', { path: 'a.txt', content: 'x' })).ok, true);
    assert.match((await agent.call('fs.write', { path: 'b.txt', content: 'x' })).reason, /write budget 1 already spent/);
    assert.match((await agent.call('fs.delete', { path: 'a.txt' })).reason, /delete budget 0 already spent/);
    agent.policy.startRun();
    agent.policy.set({ max_tool_calls: 2, max_writes: 20 });
    // the call budget gates mutating tools only, so probe with one of those
    assert.strictEqual((await agent.call('context.push', { role: 'user', content: '1' })).ok, true);
    assert.strictEqual((await agent.call('context.push', { role: 'user', content: '2' })).ok, true);
    const third = await agent.call('context.push', { role: 'user', content: '3' });
    assert.strictEqual(third.blocked, true);
    assert.match(third.reason, /tool-call budget 2 already spent/);
    assert.strictEqual((await agent.call('echo', { text: 'reads are free' })).ok, true);
  });

  test('dry-run refuses mutating tools only', async () => {
    const agent = makeAgent({ dry_run: true });
    assert.match((await agent.call('context.push', { role: 'user', content: 'x' })).reason, /dry-run refuses mutating/);
    assert.strictEqual((await agent.call('echo', { text: 'x' })).ok, true);
  });

  test('forbidden command patterns are refused even for an allowed tool', async () => {
    const agent = makeAgent();
    agent.policy.set({ confirm_required: [], deny: ['agent.run', 'fs.delete'] });
    const [ok, reason] = agent.policy.check(agent.registry.get('shell.run'), { cmd: 'sudo rm -rf /tmp/x' });
    assert.strictEqual(ok, false);
    assert.match(reason, /is forbidden/);
    assert.match((await agent.call('shell.run', { cmd: 'shutdown now' })).reason, /is forbidden/);
    assert.strictEqual(agent.policy.check(agent.registry.get('shell.run'), { cmd: 'node -v' })[0], true);
  });

  test('wall-clock limit blocks once exceeded', async () => {
    const agent = makeAgent();
    agent.policy.set({ max_seconds: 5 });
    agent.policy.startRun();
    agent.policy.started_at = Date.now() / 1000 - 10;   // the policy reads the module clock
    const [ok, reason] = agent.policy.check(agent.registry.get('echo'), {});
    assert.strictEqual(ok, false);
    assert.match(reason, /wall-clock limit 5s/);
  });

  test('unknown policy knobs are rejected, and promptBlock summarises', async () => {
    const agent = makeAgent();
    assert.throws(() => agent.policy.set({ nonsense: 1 }), /no such policy knob 'nonsense'/);
    assert.match((await agent.call('policy.set', { nonsense: 1 })).error, /policy\.set got unexpected arg 'nonsense'/);
    const block = agent.policy.promptBlock();
    assert.match(block, /root=/);
    assert.match(block, /network=off/);
    assert.match(block, /needs approval=/);
  });

  test('policy.check answers without spending budget', async () => {
    const agent = makeAgent();
    const outcome = await agent.call('policy.check', { tool: 'context.push', args: { role: 'user', content: 'x' } });
    assert.strictEqual(outcome.value.allowed, true);
    assert.strictEqual(outcome.value.mutating, true);
    assert.strictEqual(agent.policy.calls, 1);            // only policy.check itself
    assert.strictEqual(agent.policy.writes, 0);           // the probed mutation was never charged
    const blocked = (await agent.call('policy.check', { tool: 'fs.delete', args: { path: 'x' } })).value;
    assert.strictEqual(blocked.allowed, false);
    assert.match(blocked.reason, /denied by guardrail rule 'fs\.delete'/);
    assert.strictEqual(blocked.danger, true);
    assert.strictEqual(agent.policy.blocks.length, 0);    // a probe leaves no block behind
    assert.strictEqual(agent.policy.calls, 2);
  });
});

// ---------------------------------------------------------------------------
suite('built-in tools', (test) => {
  const withApp = () => {
    const agent = makeAgent();
    fs.writeFileSync(path.join(agent.root, 'app.py'), 'def add(a, b):\n    return a - b\n');
    return agent;
  };

  test('fs.edit patches one unique match', async () => {
    const agent = withApp();
    agent.grant('fs.edit', true);
    const record = await agent.call('fs.edit', { path: 'app.py', find: 'return a - b', replace: 'return a + b' });
    assert.ok(record.ok);
    assert.strictEqual(record.value.replacements, 1);
    assert.match(fs.readFileSync(path.join(agent.root, 'app.py'), 'utf8'), /a \+ b/);
  });

  test('fs.edit refuses an ambiguous needle unless all is set', async () => {
    const agent = withApp();
    agent.grant('fs.edit', true);
    fs.writeFileSync(path.join(agent.root, 'dup.py'), 'x = 1\nx = 1\n');
    const vague = await agent.call('fs.edit', { path: 'dup.py', find: 'x = 1', replace: 'y = 2' });
    assert.match(vague.error, /matches 2 times/);
    const absent = await agent.call('fs.edit', { path: 'dup.py', find: 'gone', replace: 'z' });
    assert.strictEqual(absent.error, 'no match for gone in dup.py');
    const every = await agent.call('fs.edit', { path: 'dup.py', find: 'x = 1', replace: 'y = 2', all: true });
    assert.strictEqual(every.value.replacements, 2);
  });

  test('fs.edit awaits approval and is charged as a write', async () => {
    const agent = withApp();
    assert.ok(agent.policy.confirm_required.includes('fs.edit'));
    const blocked = await agent.call('fs.edit', { path: 'app.py', find: 'a - b', replace: 'a + b' });
    assert.ok(blocked.blocked);
    assert.match(blocked.reason, /needs approval/);
    assert.match(fs.readFileSync(path.join(agent.root, 'app.py'), 'utf8'), /a - b/);
  });

  test('fs.edit stays inside the sandbox', async () => {
    const agent = withApp();
    agent.grant('fs.edit', true);
    const record = await agent.call('fs.edit', { path: '../elsewhere.py', find: 'a', replace: 'b' });
    assert.ok(record.blocked);
    assert.match(record.reason, /escapes the sandbox/);
  });

  test('browser.open is a network tool that reduces a page', async () => {
    const agent = makeAgent();
    const page = '<html><head><title>Notes &amp; more</title><style>p{color:red}</style></head>'
      + '<body><script>secret()</script><p>one   funnel</p><a href="/next">Next</a></body></html>';
    const url = `data:text/html;charset=utf-8,${encodeURIComponent(page)}`;
    const refused = await agent.call('browser.open', { url });
    assert.ok(refused.blocked);
    assert.match(refused.reason, /network tools are off/);
    agent.policy.set({ allow_network: true });
    const seen = (await agent.call('browser.open', { url })).value;
    assert.strictEqual(seen.status, 200);
    assert.strictEqual(seen.title, 'Notes & more');
    assert.strictEqual(seen.text, 'one funnel Next');
    assert.deepStrictEqual(seen.links, [{ href: '/next', text: 'Next' }]);
  });

  test('shell.run needs its deny rule lifted and an approval', async () => {
    const agent = makeAgent();
    const cmd = `${JSON.stringify(process.execPath)} --version`;
    const denied = await agent.call('shell.run', { cmd });
    assert.ok(denied.blocked);
    assert.match(denied.reason, /guardrail rule/);
    agent.policy.set({ deny: ['agent.run', 'fs.delete'] });
    const unapproved = await agent.call('shell.run', { cmd });
    assert.match(unapproved.reason, /needs approval/);
    agent.grant('shell.run', true);
    const ran = await agent.call('shell.run', { cmd });
    assert.ok(ran.ok);
    assert.strictEqual(ran.value.code, 0);
    assert.match(ran.value.stdout, /^v\d+/);
    const forbidden = await agent.call('shell.run', { cmd: 'sudo npm publish' });
    assert.ok(forbidden.blocked);
    assert.match(forbidden.reason, /is forbidden/);
  });
});

suite('hooks', (test) => {
  test('a pre_tool hook can block a call', async () => {
    const agent = makeAgent();
    agent.hook('pre_tool', 'fs.*', (payload) => (String(payload.args.path).includes('secret')
      ? { block: true, reason: 'private' } : null));
    assert.strictEqual((await agent.call('fs.read', { path: 'secret.txt' })).reason, 'private');
    assert.strictEqual((await agent.call('fs.read', { path: 'open.txt' })).blocked, false);
  });

  test('a pre_tool hook can rewrite args', async () => {
    const agent = makeAgent();
    agent.hook('pre_tool', 'echo', () => ({ args: { text: 'rewritten' } }));
    assert.strictEqual((await agent.call('echo', { text: 'original' })).value.echo, 'rewritten');
  });

  test('a post_tool veto turns a success into a block', async () => {
    const agent = makeAgent();
    agent.hook('post_tool', 'echo', () => ({ block: true, reason: 'not on the transcript' }));
    const record = await agent.call('echo', { text: 'x' });
    assert.strictEqual(record.ok, false);
    assert.match(record.reason, /post_tool veto: not on the transcript/);
  });

  test('a broken hook blocks safely instead of crashing the loop', async () => {
    const agent = makeAgent();
    agent.hook('pre_tool', 'echo', () => { throw new Error('hook bug'); });
    const record = await agent.call('echo', { text: 'x' });
    assert.strictEqual(record.blocked, true);
    assert.match(record.reason, /hook .* raised Error: hook bug/);
  });

  test('a hook handler can be a registered tool', async () => {
    const agent = makeAgent();
    const seen = [];
    agent.registry.add(new Tool({ name: 'audit.record', desc: 'record the tool name', args: { payload: 'object' },
                                  kind: 'hook', handler: async ({ payload }) => { seen.push(payload.tool); return null; } }));
    agent.hook('pre_tool', '*', null, 'audit.record');
    const record = await agent.call('echo', { text: 'x' });
    assert.strictEqual(record.ok, true);
    assert.deepStrictEqual(seen, ['echo']);
    assert.strictEqual(agent.hooks.fired.at(-1).hook, 'audit.record');
  });

  test('hook events are validated, and firings are recorded', async () => {
    const agent = makeAgent();
    assert.throws(() => agent.hook('nope', '*', () => null), ToolError);
    assert.throws(() => agent.hook('pre_tool', '*', null, 'audit.missing'), ToolError);
    assert.throws(() => agent.hook('pre_tool', '*'), ToolError);
    agent.hook('pre_tool', 'echo', () => null, '', 'watch');
    agent.hook('post_tool', 'echo', () => null, '', 'witness');
    await agent.call('echo', { text: 'x' });
    const events = new Set(agent.hooks.fired.map((row) => row.event));
    assert.ok(events.has('pre_tool') && events.has('post_tool'));
    const listed = await agent.call('hook.list', {});
    assert.deepStrictEqual(listed.value, [{ name: 'watch', event: 'pre_tool', pattern: 'echo', tool: '' },
                                          { name: 'witness', event: 'post_tool', pattern: 'echo', tool: '' }]);
    const history = await agent.call('hook.history', { limit: 2 });
    assert.strictEqual(history.value.length, 2);
  });

  test('hook.remove detaches by name', async () => {
    const agent = makeAgent();
    agent.hook('pre_tool', '*', () => null, '', 'guard');
    assert.strictEqual((await agent.call('hook.remove', { name: 'guard' })).value, 1);
    assert.strictEqual(agent.hooks.hooks.length, 0);
    assert.strictEqual((await agent.call('hook.remove', { name: 'guard' })).value, 0);
  });
});
// ---------------------------------------------------------------------------
suite('memory', (test) => {
  test('write, read, search and forget', async () => {
    const memory = new Memory(path.join(workspace(), 'memory.json'));
    memory.write({ key: 'route', value: 'walk the funnel', tags: 'core', kind: 'fact' });
    memory.write({ key: 'other', value: 'unrelated' });
    assert.strictEqual(memory.read({ key: 'route' }).value, 'walk the funnel');
    assert.strictEqual(memory.search({ query: 'funnel' })[0].key, 'route');
    assert.strictEqual(memory.search({ query: 'route', kind: 'note' }).length, 0);
    assert.deepStrictEqual(memory.forget({ key: 'other' }), { forgotten: 1, left: 1 });
    assert.match(memory.notes(), /- route \(fact\): walk the funnel/);
  });

  test('notes survive a new instance when a path is given', async () => {
    const file = path.join(workspace(), 'nested', 'memory.json');
    new Memory(file).write({ key: 'k', value: 'v' });
    assert.strictEqual(new Memory(file).entries.length, 1);
    assert.strictEqual(new Memory(null).entries.length, 0);
  });

  test('memory is reachable through the agent funnel', async () => {
    const agent = makeAgent({ memory_path: path.join(workspace(), 'memory.json') });
    assert.strictEqual((await agent.call('memory.write', { key: 'k', value: 'v' })).ok, true);
    assert.strictEqual((await agent.call('memory.search', { query: 'k' })).value[0].value, 'v');
    assert.strictEqual((await agent.call('memory.list', {})).value.length, 1);
    assert.match((await agent.call('memory.read', { key: 'nope' })).error, /no memory entry/);
    assert.strictEqual((await agent.call('memory.forget', { key: 'k' })).value.forgotten, 1);
  });
});

// ---------------------------------------------------------------------------
suite('context', (test) => {
  test('stats count tokens against the budget', () => {
    const context = new Context({ budget_tokens: 8 });
    context.reset('system prompt');
    context.push('user', 'hello there this is a goal');
    const stats = context.stats();
    assert.strictEqual(stats.messages, 1);
    assert.strictEqual(stats.tokens, 9);
    assert.strictEqual(stats.over, true);
    assert.strictEqual(stats.used_pct, 112.5);
  });

  test('compact folds only the old turns', () => {
    const context = new Context({ keep_last: 2 });
    for (let index = 0; index < 6; index += 1) context.push('tool', `observation ${index} ${'x'.repeat(80)}`, `t${index}`);
    const outcome = context.compact();
    assert.strictEqual(outcome.compacted, true);
    assert.strictEqual(outcome.folded, 4);
    assert.ok(outcome.after < outcome.before);
    assert.strictEqual(context.messages.at(-1).name, 't5');
    assert.strictEqual(new Context({ keep_last: 6 }).compact().compacted, false);
  });

  test('a summary can be supplied, and names ride along', () => {
    const context = new Context({ keep_last: 1 });
    for (let index = 0; index < 4; index += 1) context.push('tool', `turn ${index}`);
    context.compact('the user wanted a report');
    assert.match(context.messages[0].content, /the user wanted a report/);
    const fresh = new Context();
    fresh.reset('sys');
    fresh.push('tool', 'out', 'fs.read');
    assert.deepStrictEqual(fresh.asMessages(), [{ role: 'system', content: 'sys' },
      { role: 'tool', content: 'out', name: 'fs.read' }]);
  });

  test('context tools see the same window', async () => {
    const agent = makeAgent();
    await agent.call('context.push', { role: 'user', content: 'a thing happened', name: 'note' });
    assert.strictEqual((await agent.call('context.stats', {})).value.messages, 1);
    assert.strictEqual((await agent.call('context.read', { limit: 1 })).value[0].name, 'note');
    assert.strictEqual((await agent.call('context.compact', {})).value.compacted, false);
  });
});

// ---------------------------------------------------------------------------
suite('prompts', (test) => {
  test('rendered prompts carry every layer that matters', async () => {
    const agent = makeAgent({ memory_path: path.join(workspace(), 'memory.json') });
    agent.memory.write({ key: 'known', value: 'a fact' });
    const text = agent.prompts.render(agent);
    assert.match(text, /acts only through tools/);
    assert.match(text, /# memory\n- known/);
    assert.match(text, /# guardrails\nroot=/);
    assert.match(text, /# tools\n/);
    assert.match(text, /echo <text>/);
  });

  test('denied tools are hidden from the tool block', () => {
    const agent = makeAgent();
    const toolsBlock = agent.prompts.render(agent).split('# tools\n')[1];
    for (const hidden of ['fs.delete', 'shell.run', 'agent.run']) assert.ok(!toolsBlock.includes(hidden), hidden);
    assert.ok(toolsBlock.includes('echo'));
    agent.policy.set({ deny: [] });
    assert.ok(agent.toolPrompt().includes('fs.delete'));
  });

  test('layers can be replaced and recomputed', async () => {
    const agent = makeAgent();
    const outcome = await agent.call('prompt.set', { layer: 'goal', text: 'ship it' });
    assert.strictEqual(outcome.value.tokens, estimateTokens('ship it'));
    agent.prompts.attach('tasks', () => JSON.stringify(agent.tasks.summary()));
    agent.tasks.create({ subject: 'do the thing' });
    assert.match(agent.prompts.render(agent), /"tasks":1/);
    assert.match((await agent.call('prompt.set', { layer: 'nope', text: 'x' })).error, /unknown prompt layer/);
    const got = await agent.call('prompt.get', { rendered: true });
    assert.ok(got.value.rendered.length > 100);
    assert.deepStrictEqual(got.value.dynamic, ['tasks']);
  });

  test('the persona layer is the one the base prompt promises', () => {
    assert.match(H.PERSONA_PROMPT, /Never invent a path or a tool name/);
    assert.match(H.BASE_PROMPT, /\{"thought": "<one line>", "final"/);
  });
});

// ---------------------------------------------------------------------------
suite('skills', (test) => {
  const packFile = (root, text) => {
    const dir = path.join(root, 'skills');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'pack.md'), text);
    return dir;
  };

  test('front matter is parsed into name, description, when and tools', async () => {
    const root = workspace();
    const skills = new Skills([packFile(root, '---\nname: notekeeper\ndescription: keep notes\nwhen_to_use: when writing\n'
      + 'tools: fs.read, fs.write, nope.tool\n---\nRead first, then write.\n')], { registry: makeAgent().registry });
    const listed = skills.list();
    assert.strictEqual(listed[0].name, 'notekeeper');
    assert.strictEqual(listed[0].when, 'when writing');
    assert.deepStrictEqual(listed[0].tools, ['fs.read', 'fs.write', 'nope.tool']);
    assert.strictEqual(skills.pack('notekeeper').body, 'Read first, then write.');
  });

  test('activating a skill joins the prompt and reports missing tools', async () => {
    const agent = makeAgent();
    const dir = packFile(agent.root, '---\nname: notekeeper\ntools: fs.read, nope.tool\n---\nRead notes.md first.\n');
    agent.skills = new Skills([dir], { registry: agent.registry });
    agent.skills.register(agent.registry);
    const outcome = await agent.call('skills.use', { name: 'notekeeper' });
    assert.deepStrictEqual(outcome.value.missing_tools, ['nope.tool']);
    assert.match(agent.prompts.render(agent), /## skill: notekeeper/);
    await agent.call('skills.stop', { name: 'notekeeper' });
    assert.ok(!agent.prompts.render(agent).includes('## skill:'));
    assert.match((await agent.call('skills.use', { name: 'ghost' })).error, /no skill 'ghost'/);
    assert.strictEqual((await agent.call('skills.scan', {})).value.found.length, 1);
  });

  test('a body with no front matter is still a skill', () => {
    const skills = new Skills([packFile(workspace(), 'plain instructions\n')], { registry: makeAgent().registry });
    assert.strictEqual(skills.list().length, 1);
    assert.strictEqual(skills.list()[0].name, 'pack');
  });
});

// ---------------------------------------------------------------------------
suite('tasks and scheduler', (test) => {
  test('tasks track status and blockers', async () => {
    const agent = makeAgent();
    const first = await agent.call('task.create', { subject: 'extract' });
    const second = await agent.call('task.create', { subject: 'write', blocks: String(first.value.id) });
    assert.deepStrictEqual(second.value.blocked_by, [1]);
    assert.strictEqual((await agent.call('task.ready', {})).value.length, 1);
    await agent.call('task.update', { id: 1, status: 'completed' });
    assert.deepStrictEqual((await agent.call('task.ready', {})).value.map((task) => task.subject), ['write']);
    assert.strictEqual((await agent.call('task.update', { id: 1, status: 'nonsense' })).error,
      "bad task status 'nonsense'; use " + H.TASK_STATUSES.join(', '));
    assert.match((await agent.call('task.get', { id: 99 })).error, /no task 99/);
    const listed = await agent.call('task.list', {});
    assert.strictEqual(listed.value.completed, 1);
    assert.deepStrictEqual(listed.value.ready, ['write']);
  });

  test('a job is a tool call through the same funnel', async () => {
    const agent = makeAgent();
    agent.scheduler.add({ name: 'beat', tool: 'echo', args: { text: 'tick' }, every: 1, max_runs: 2 });
    assert.strictEqual(agent.scheduler.due(agent.clock() + 0.5).length, 0);
    const fired = await agent.scheduler.tick((name, args) => agent.dispatch(name, args), agent.clock() + 2);
    assert.strictEqual(fired.length, 1);
    assert.strictEqual(fired[0].result.result.echo, 'tick');
    assert.strictEqual(agent.scheduler.jobs[0].runs, 1);
    await agent.scheduler.tick((name, args) => agent.dispatch(name, args), agent.clock() + 4);
    await agent.scheduler.tick((name, args) => agent.dispatch(name, args), agent.clock() + 6);
    assert.strictEqual(agent.scheduler.jobs[0].runs, 2);          // max_runs is respected
    assert.deepStrictEqual(agent.scheduler.due(agent.clock() + 10), []);
    assert.deepStrictEqual((await agent.call('schedule.list', {})).value.jobs.length, 1);
    assert.deepStrictEqual((await agent.call('schedule.remove', { name: 'beat' })).value, { removed: 1, left: 0 });
  });

  test('jobs must name a tool and a schedule', () => {
    const scheduler = new Scheduler();
    assert.throws(() => scheduler.add({ name: 'x' }), ToolError);
    assert.throws(() => scheduler.add({ name: 'x', tool: 'echo' }), ToolError);
  });

  test('schedule.add is validated through the funnel too', async () => {
    const agent = makeAgent();
    assert.match((await agent.call('schedule.add', { name: 'x', tool: 'echo' })).error, /every=<seconds> or at=<epoch>/);
    assert.match((await agent.call('schedule.add', { name: 'y', tool: '', every: 60 })).error, /must name a tool/);
    assert.deepStrictEqual((await agent.call('schedule.due', {})).value, { due: [] });
  });

  test('the loop ticks whatever is due', async () => {
    const agent = makeAgent();
    agent.scheduler.add({ name: 'beat', tool: 'echo', args: { text: 'tick' }, at: 1.0 });
    agent.script([{ call: { tool: 'time.now', args: {} } }, { final: 'done' }]);
    const run = await agent.run('let the loop tick');
    assert.strictEqual(run.status, 'done');
    assert.deepStrictEqual(agent.scheduler.fired.map((row) => row.job), ['beat']);
  });
});

// ---------------------------------------------------------------------------
suite('models', (test) => {
  test('model.set_script queues actions for model.scripted', async () => {
    const agent = makeAgent();
    const outcome = await agent.call('model.set_script', { turns: [{ final: 'queued' }] });
    assert.strictEqual(outcome.value.turns, 1);
    assert.strictEqual((await agent.call('model.scripted', {})).value.final, 'queued');
    const exhausted = await agent.call('model.scripted', {});
    assert.match(exhausted.value.final, /script 'default' ran out after 1 turns/);
  });

  test('named scripts keep separate queues and can be appended to', () => {
    const models = new Models({ registry: new Registry() });
    models.script({ script: 'a', turns: [{ final: '1' }] });
    models.script({ script: 'b', turns: [{ final: '2' }], append: true });
    models.script({ script: 'a', turns: [{ final: '3' }], append: true });
    assert.strictEqual(models.scripted({ script: 'a' }).final, '1');
    assert.strictEqual(models.scripted({ script: 'a' }).final, '3');
    assert.strictEqual(models.scripted({ script: 'b' }).final, '2');
  });

  test('endpoint replies are normalised to the contract', () => {
    const models = new Models({ registry: new Registry() });
    assert.deepStrictEqual(models.actionFrom('{"thought": "t", "final": "f"}'), { thought: 't', final: 'f' });
    assert.deepStrictEqual(models.actionFrom('noise {"action": {"tool": "echo", "args": {}}} noise'),
      { call: { tool: 'echo', args: {} } });
    assert.deepStrictEqual(models.actionFrom('called it', [{ function: { name: 'echo', arguments: '{"text":"x"}' } }]),
      { thought: 'called it', call: { tool: 'echo', args: { text: 'x' } } });
    assert.throws(() => models.actionFrom('no json here'), ToolError);
  });

  test('local endpoints are ordinary networked tools with their own args', () => {
    const agent = makeAgent();
    for (const name of ['model.ollama', 'model.openai', 'model.llamacpp']) {
      const spec = agent.registry.get(name);
      assert.strictEqual(spec.network, true, `${name} should be a network tool`);
      for (const shared of ['messages', 'tools', 'goal', 'step']) assert.ok(shared in spec.args, `${name} lacks ${shared}`);
    }
    assert.ok('url' in agent.registry.get('model.ollama').args);
    assert.ok('base_url' in agent.registry.get('model.openai').args);
    assert.ok('n_predict' in agent.registry.get('model.llamacpp').args);
  });

  test('the scripted model needs no network and no key', async () => {
    const agent = makeAgent();
    const call = await agent.call('model.scripted', { messages: [{ role: 'user', content: 'hi' }], tools: ['echo'], goal: 'g', step: 0 });
    assert.strictEqual(call.ok, true);
    assert.strictEqual(agent.policy.summary().used.calls, 1);
  });

  test('chatml rendering is what llamacpp would send', () => {
    const { CHATML_START, CHATML_MESSAGE, CHATML_END } = H;
    assert.strictEqual(H.renderChatml([{ role: 'system', content: 'hi' }]),
      CHATML_START + 'system' + CHATML_MESSAGE + 'hi' + CHATML_END + CHATML_START + 'assistant' + CHATML_MESSAGE);
    const two = H.renderChatml([{ role: 'system', content: 'a' }, { role: 'user', content: 'b' }]);
    assert.ok(two.includes('user' + CHATML_MESSAGE + 'b' + CHATML_END));
    assert.ok(!two.includes('system: a'));   // the old plain-role template is gone
  });

  test('llamacpp posts the model name a router-mode server demands', async () => {
    let seen = null;
    const server = http.createServer((req, res) => {
      let raw = '';
      req.on('data', (chunk) => { raw += chunk; });
      req.on('end', () => {
        seen = JSON.parse(raw);
        res.writeHead(200, { 'content-type': 'application/json' });
        // a real server hands back the reply plus the terminator it stopped on
        res.end(JSON.stringify({ content: '{"thought": "48", "final": "48"}' + H.CHATML_END }));
      });
    });
    await new Promise((done) => server.listen(0, '127.0.0.1', done));
    const agent = makeAgent();
    agent.policy.set({ allow_network: true });
    try {
      const record = await agent.call('model.llamacpp', {
        base_url: `http://127.0.0.1:${server.address().port}`, model: 'tiny:Q4_K_M',
        messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: '12 * 4' },
                   { role: 'user', content: 'observation: echo said hi' }], goal: '12 * 4' });
      assert.ok(record.ok, record.error);
      assert.deepStrictEqual(record.value, { thought: '48', final: '48' });
      assert.strictEqual(seen.model, 'tiny:Q4_K_M');
      assert.ok(seen.prompt.startsWith(H.CHATML_START));
      assert.ok(seen.prompt.includes('observation: echo said hi'));  // the window is not dropped
      assert.deepStrictEqual(seen.stop, H.CHATML_STOPS);
      // a stringified schema is a 400 on llama.cpp: it wants the object itself
      assert.strictEqual(typeof seen.json_schema, 'object');
      assert.deepStrictEqual(seen.json_schema.required, ['thought']);
    } finally { server.close(); }
  });
});

// ---------------------------------------------------------------------------
suite('loop', (test) => {
  test('the window carries the action, not just its thought', async () => {
    // a completion endpoint continues from the assistant turn: if the window only
    // holds the thought, the model repeats its last tool call forever
    const agent = makeAgent();
    agent.script([{ thought: 'say hi', call: { tool: 'echo', args: { text: 'hi' } } },
                  { thought: 'done', final: 'echo said hi' }]);
    const run = await agent.run('echo once');
    assert.strictEqual(run.status, 'done');
    const assistant = agent.context.asMessages().filter((m) => m.role === 'assistant');
    assert.ok(assistant[0].content.includes('"call"'));
    assert.ok(assistant[0].content.includes('echo'));
  });

  test('steps run out rather than spinning', async () => {
    const agent = makeAgent();
    agent.script([{ call: { tool: 'echo', args: { text: 'again' } } }]);
    const run = await agent.run('loop forever', { max_steps: 3 });
    assert.strictEqual(run.status, 'done');               // the script runs out and says so
    assert.match(run.answer, /ran out after/);
    agent.script([1, 2, 3, 4, 5].map(() => ({ call: { tool: 'echo', args: { text: 'again' } } })));
    const spinning = await agent.run('loop forever', { max_steps: 3 });
    assert.strictEqual(spinning.status, 'steps-exhausted');
    assert.strictEqual(spinning.turns.length, 3);
  });

  test('two model failures in a row stop the run', async () => {
    const agent = makeAgent();
    agent.model = 'model.missing';
    const run = await agent.run('no brain');
    assert.strictEqual(run.status, 'model-failed');
    assert.match(run.error, /no tool 'model.missing'/);
  });

  test('the window compacts itself while the loop runs', async () => {
    const agent = makeAgent({ budget_tokens: 120, keep_last: 2 });
    agent.script([{ call: { tool: 'fs.read', args: { path: 'big.txt' } } },
                  { call: { tool: 'echo', args: { text: 'x'.repeat(400) } } }, { final: 'done' }]);
    fs.writeFileSync(path.join(agent.root, 'big.txt'), 'y'.repeat(600));
    const run = await agent.run('fill the window');
    assert.ok(run.compactions >= 1);
    assert.ok(agent.context.compactions >= 1);
  });

  test('agent.step drives one cycle at a time', async () => {
    const agent = makeAgent();
    agent.script([{ call: { tool: 'echo', args: { text: 'one' } } }]);
    const outcome = await agent.step('stepwise');
    assert.strictEqual(outcome.ok, true);
    assert.strictEqual(outcome.call.tool, 'echo');
    assert.strictEqual(agent.runs.length, 0);
  });

  test('agent.run is denied by default, and says so in the prompt', async () => {
    const agent = makeAgent();
    const record = await agent.call('agent.run', { goal: 'self drive' });
    assert.strictEqual(record.blocked, true);
    assert.match(agent.prompts.render(agent), /# guardrails/);
  });

  test('status and panel agree about the live picture', async () => {
    const agent = makeAgent();
    const status = await agent.call('agent.status', {});
    const panel = agent.panel();
    assert.deepStrictEqual(Object.keys(status.value), Object.keys(panel));
    assert.strictEqual(panel.tools, 57);
    assert.strictEqual(panel.model, 'model.scripted');
    assert.deepStrictEqual(Object.keys(panel.kinds).sort(),
      ['context', 'hook', 'loop', 'memory', 'model', 'policy', 'prompt', 'schedule', 'skill', 'task', 'tool']);
  });

  test('the demo run blocks exactly the secret read and the delete', async () => {
    const root = workspace();
    fs.writeFileSync(path.join(root, 'secret.txt'), 'do not read');
    const agent = new Agent({ name: 'demo', root, memory_path: path.join(root, '.agent', 'memory.json'), budget_tokens: 900 });
    agent.hook('pre_tool', 'fs.*', (payload) => (String((payload.args || {}).path || '').includes('secret')
      ? { block: true, reason: 'the no-secrets hook refuses private files' } : null), '', 'no-secrets');
    agent.grant('fs.write');
    agent.script([
      { call: { tool: 'fs.write', args: { path: 'notes.md', content: '# notes\nalpha beta gamma delta' } } },
      { call: { tool: 'fs.read', args: { path: 'notes.md' } } },
      { call: { tool: 'fs.read', args: { path: 'secret.txt' } } },
      { call: { tool: 'fs.delete', args: { path: 'notes.md' } } },
      { memory: { key: 'note-length', value: '4 words', kind: 'fact' }, task: { subject: 'expand the notes' } },
      { call: { tool: 'memory.search', args: { query: 'note' } } },
      { final: 'notes.md holds 4 words; secret read and delete were both blocked' },
    ]);
    const run = await agent.run('summarise the notes and prove the guardrails bite');
    assert.strictEqual(run.trail, 'fs.write -> fs.read -> fs.read -> fs.delete -> memory.search');
    assert.deepStrictEqual(run.blocked.map((row) => row.tool), ['fs.read', 'fs.delete']);
    assert.strictEqual(run.answer, 'notes.md holds 4 words; secret read and delete were both blocked');
    assert.ok(agent.context.compactions >= 1);
  });
});

// ---------------------------------------------------------------------------
suite('plugins', (test) => {
  test('a file in the tools directory becomes a named, guardrailed tool', async () => {
    const dir = workspace();
    fs.writeFileSync(path.join(dir, 'wordcount.js'), "'use strict';\n"
      + `const { tool } = require(${JSON.stringify(path.join(__dirname, 'agent.js'))});\n`
      + 'module.exports = tool(\'words.count\', \'count the words in the text\',\n'
      + "  { args: { text: 'string' }, required: ['text'] },\n"
      + '  ({ text }) => ({ words: text.split(/\\s+/).filter(Boolean).length }));\n');
    const agent = new Agent({ root: workspace(), tools_dir: dir, skills_dir: null, memory_path: null });
    assert.deepStrictEqual(agent.plugins.loaded, ['words.count']);
    assert.deepStrictEqual(agent.plugins.errors, {});
    assert.strictEqual((await agent.call('words.count', { text: 'one two three' })).value.words, 3);
    await agent.call('policy.set', { deny: ['agent.run', 'fs.delete', 'shell.*', 'words.*'] });
    const blocked = await agent.call('words.count', { text: 'one two three' });
    assert.strictEqual(blocked.blocked, true);
    assert.match(blocked.reason, /denied by guardrail rule 'words\.\*'/);
    H.defaultRegistry.remove('words.count');       // do not leak the plug into other suites
  });

  test('entry-point style plugins are adopted with their metadata', async () => {
    const agent = makeAgent();
    const before = agent.registry.length;
    const outcome = discover(null, { registry: agent.registry,
      entryPoints: (group) => group === H.ENTRY_POINT_GROUP
        ? [{ name: 'pack', value: new Tool({ name: 'pack.echo', desc: 'from the pack', args: { text: 'string' },
                                             required: ['text'], handler: async ({ text }) => ({ echo: text }) }) }]
        : [] });
    assert.deepStrictEqual(outcome.loaded, ['pack.echo']);
    assert.strictEqual(agent.registry.length, before + 1);
    assert.strictEqual((await agent.call('pack.echo', { text: 'hi' })).value.echo, 'hi');
  });

  test('a broken plugin is reported, not thrown', async () => {
    const dir = workspace();
    fs.writeFileSync(path.join(dir, 'bad.js'), 'this is not valid javascript )(;\n');
    const agent = new Agent({ root: workspace(), tools_dir: dir, skills_dir: null, memory_path: null });
    assert.deepStrictEqual(agent.plugins.loaded, []);
    assert.match(Object.values(agent.plugins.errors)[0], /Error/);
  });

  test('tool() overloads all register something callable', async () => {
    const registry = new Registry();
    tool('a.one', 'first form', { registry, args: {}, handler: null }, async () => 'one');
    tool('a.two', { registry, args: { x: 'string' }, required: ['x'] }, async ({ x }) => x);
    assert.strictEqual(registry.length, 2);
    assert.strictEqual(await registry.get('a.one').call({}), 'one');
    assert.strictEqual(await registry.get('a.two').call({ x: 'y' }), 'y');
  });
});

// ---------------------------------------------------------------------------
suite('single file and surface', (test) => {
  test('the file needs nothing but node builtins', () => {
    const source = fs.readFileSync(path.join(__dirname, 'agent.js'), 'utf8');
    const code = source.split('\n').filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line)).join('\n');
    const requires = [...code.matchAll(/require\((['"])(.*?)\1\)/g)].map((row) => row[2]);
    assert.ok(requires.length > 0);
    for (const id of requires) {
      assert.ok(['fs', 'os', 'path', 'child_process'].includes(id) || !id.startsWith('.'), `unexpected dependency ${id}`);
    }
    assert.ok(source.length > 40000);
  });

  test('helpers behave the way the engine assumes', () => {
    assert.strictEqual(estimateTokens('Style: dry.'), 2);
    assert.strictEqual(firstJson('x {"a": {"b": "}"}} y').a.b, '}');
    assert.strictEqual(firstJson('no json'), null);
    assert.deepStrictEqual(H.splitCommand("echo 'one two' \"three four\""), ['echo', 'one two', 'three four']);
    assert.ok(under(path.resolve('a/b.txt'), process.cwd()));
    assert.ok(!under(path.resolve('../escape'), process.cwd()));
    assert.ok(resolvePath('x', process.cwd()).includes('x'));
    assert.throws(() => new Registry().get('missing'), UnknownTool);
    assert.throws(() => { throw new Blocked('x'); }, Blocked);
    assert.throws(() => { throw new SandboxError('x'); }, SandboxError);
  });

  test('every component kind has a tool surface', async () => {
    const agent = makeAgent();
    const names = new Set(agent.registry.names());
    for (const required of ['tools.list', 'skills.use', 'memory.write', 'context.compact', 'hook.add',
                            'policy.check', 'prompt.set', 'task.create', 'schedule.add', 'model.set_script', 'agent.run']) {
      assert.ok(names.has(required), `missing tool ${required}`);
    }
  });
});

// ---------------------------------------------------------------------------
suite('parity with the other two implementations', (test) => {
  const jsNames = new Set(makeAgent().registry.names());

  test('agent.py registers exactly the same tools', () => {
    const python = spawnSync('python', ['-c',
      'import json,sys,pathlib; import agent; '
      + 'print(json.dumps(sorted(agent.Agent(root=pathlib.Path("."), skills_dir=None, memory_path=None).registry.names())))'],
      { cwd: __dirname, encoding: 'utf8' });
    if (python.status !== 0) {
      console.log('    (skipped: agent.py could not be imported)');
      return;
    }
    assert.deepStrictEqual(JSON.parse(python.stdout.trim()), [...jsNames].sort());
  });

  test('agent.html registers the same tools plus the manual model', () => {
    const source = fs.readFileSync(path.join(__dirname, 'agent.html'), 'utf8');
    // only the engine section registers tools; the panel and its presets do not
    const engine = source.slice(source.indexOf('// -- engine'), source.indexOf('const PRESETS'));
    const found = [...engine.matchAll(/new Tool\(\s*\{\s*name: '([a-z_]+(?:\.[a-z_]+)*)'/g)].map((row) => row[1]);
    const htmlNames = new Set(found);
    assert.deepStrictEqual([...htmlNames].sort(), [...jsNames].concat(['model.manual']).sort());
    for (const name of ['agent.run', 'model.set_script', 'fs.write', 'fs.edit', 'browser.open', 'shell.run',
                        'memory.write', 'hook.add', 'policy.check', 'task.create', 'schedule.add',
                        'context.compact', 'skills.use', 'echo']) {
      assert.ok(htmlNames.has(name), `agent.html is missing ${name}`);
    }
  });

  test('the three twins agree on the guardrail vocabulary', () => {
    const agent = makeAgent();
    const expected = { deny: ['agent.run', 'fs.delete', 'shell.*'],
                       confirm_required: ['fs.write', 'fs.edit', 'shell.run'] };
    for (const [key, value] of Object.entries(expected)) assert.deepStrictEqual(agent.policy[key], value);
    const py = fs.readFileSync(path.join(__dirname, 'agent.py'), 'utf8');
    const html = fs.readFileSync(path.join(__dirname, 'agent.html'), 'utf8');
    for (const phrase of ["denied by guardrail rule", "needs approval: call confirm.grant", "escapes the sandbox",
                          "network tools are off", "dry-run refuses mutating tools", "is forbidden",
                          "already spent", "wall-clock limit"]) {
      assert.ok(py.includes(phrase), `agent.py lost ${phrase}`);
      assert.ok(html.includes(phrase), `agent.html lost ${phrase}`);
    }
  });

  test('every example runs offline in the twin', async () => {
    const log = [];
    const original = console.log;
    console.log = (...args) => log.push(args.join(' '));
    try {
      for (const name of Object.keys(examples)) assert.strictEqual(await runExamples([name]), 0, `${name} failed`);
      assert.strictEqual(await runExamples(['nope']), 2);
    } finally { console.log = original; }
    const joined = log.join('\n');
    assert.match(joined, /=== tool-loop/);
    assert.match(joined, /status=done/);
    assert.match(joined, /denied by guardrail rule 'words\.\*'/);
  });
});

// ---------------------------------------------------------------------------
const wanted = process.argv.slice(2);
const selected = groups.filter(([name]) => !wanted.length || wanted.some((term) => name.includes(term)));

async function main() {
  let failures = 0;
  let count = 0;
  for (const [name, cases] of selected) {
    console.log(`\n${name}`);
    for (const [label, fn] of cases) {
      count += 1;
      try {
        await fn();
        console.log(`  ok   ${label}`);
      } catch (error) {
        failures += 1;
        console.log(`  FAIL ${label}\n       ${(error && error.message || '').split('\n').join('\n       ')}`);
      }
    }
  }
  console.log(`\n${count - failures}/${count} passed${wanted.length ? ` (filtered by ${wanted.join(',')})` : ''}`);
  process.exitCode = failures ? 1 : 0;
}

main();

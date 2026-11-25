// Render the LSS canvas with the React and ReactDOM scripts loaded by page.py.
(() => {
  'use strict';

  const events = {
    onclick: 'onClick',
    onmouseenter: 'onMouseEnter',
    onmouseleave: 'onMouseLeave',
    onmousemove: 'onMouseMove',
    oncontextmenu: 'onContextMenu',
    onscroll: 'onScroll',
  };
  const path = '[\\w$]+(?:\\.[\\w$]+)*';
  const wholeBinding = new RegExp(`^\\s*{{\\s*(${path})\\s*}}\\s*$`);
  const bindings = new RegExp(`{{\\s*(${path})\\s*}}`, 'g');
  const css = document.createElement('div').style;

  function lookup(scope, name) {
    return name.split('.').reduce((value, part) => value?.[part], scope);
  }

  function interpolate(source, scope) {
    return source.replace(bindings, (_, name) => String(lookup(scope, name) ?? ''));
  }

  function bind(source) {
    const whole = wholeBinding.exec(source);
    return whole ? scope => lookup(scope, whole[1]) : scope => interpolate(source, scope);
  }

  function declarations(source) {
    // Let the browser parse CSS and serialize shorthands (including var()).
    // A semicolon inside a function or quoted string does not end a declaration.
    css.cssText = source;
    const serialized = css.cssText;
    const result = [];
    let start = 0;
    let depth = 0;
    for (const token of serialized.matchAll(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\\.|[();]/g)) {
      if (token[0] === '(') depth += 1;
      if (token[0] === ')') depth -= 1;
      if (token[0] !== ';' || depth) continue;
      const declaration = serialized.slice(start, token.index);
      const colon = declaration.indexOf(':');
      result.push([declaration.slice(0, colon).trim(), declaration.slice(colon + 1).trim()]);
      start = token.index + 1;
    }
    return result;
  }

  function inlineStyle(value) {
    if (typeof value !== 'string') return value;
    return Object.fromEntries(declarations(value).map(([name, declaration]) => {
      const prop = name.startsWith('--') ? name : name.replace(/^-ms-/, 'ms-')
        .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      return [prop, declaration];
    }));
  }

  function compileChildren(parent, rules) {
    const renderers = Array.from(parent.childNodes, node => compile(node, rules)).filter(Boolean);
    return scope => renderers.map(render => render(scope));
  }

  function compile(node, rules) {
    if (node.nodeType === Node.TEXT_NODE) {
      const source = node.data;
      return scope => interpolate(source, scope);
    }
    if (node.nodeType !== Node.ELEMENT_NODE || node.localName === 'helmet') return null;

    const tag = node.localName;
    const children = compileChildren(node, rules);
    if (tag === 'sc-if') {
      const value = bind(node.getAttribute('value'));
      return scope => value(scope) ? children(scope) : null;
    }
    if (tag === 'sc-for') {
      const list = bind(node.getAttribute('list'));
      const name = node.getAttribute('as');
      return scope => {
        const items = list(scope);
        if (!Array.isArray(items)) return null;
        return items.flatMap(item => children(Object.assign(Object.create(scope), { [name]: item })));
      };
    }

    const attributes = [];
    for (const { name, value } of node.attributes) {
      if (name.startsWith('hint-placeholder-') || name === 'style-hover' || name === 'style-active') continue;
      const resolve = bind(value);
      if (name === 'style') {
        const style = inlineStyle(value);
        attributes.push([name, value.includes('{{') ? scope => inlineStyle(resolve(scope)) : () => style]);
      } else if (Object.hasOwn(events, name) || name === 'ref') {
        attributes.push([events[name] ?? name, scope => {
          const handler = resolve(scope);
          return typeof handler === 'function' ? handler : undefined;
        }]);
      } else {
        attributes.push([name === 'class' ? 'className' : name, resolve]);
      }
    }

    const cssId = rules.length;
    for (const pseudo of ['hover', 'active']) {
      const style = node.getAttribute(`style-${pseudo}`);
      if (!style) continue;
      const important = declarations(style).map(([name, value]) =>
        `${name}:${value.replace(/\s*!important$/i, '')}!important;`).join('');
      rules.push(`[data-lss-pseudo="${cssId}"]:${pseudo}{${important}}`);
    }
    if (rules.length > cssId) attributes.push(['data-lss-pseudo', () => cssId]);

    return scope => {
      const props = Object.fromEntries(attributes.map(([name, resolve]) => [name, resolve(scope)]));
      return React.createElement(tag, props, children(scope));
    };
  }

  function mount() {
    const canvas = document.querySelector('x-dc');
    const script = document.querySelector('script[type="text/x-dc"][data-dc-script][data-props]');
    const rules = [];
    const render = compileChildren(canvas, rules);

    class DCLogic extends React.Component {
      state = {};

      // The canvas and the bridge read this.state right after setState, as the exported
      // runtime allowed: merge at once, then let React render the same patch.
      setState(update, callback) {
        const patch = typeof update === 'function' ? update(this.state, this.props) : update;
        if (patch != null) this.state = Object.assign({}, this.state, patch);
        super.setState(patch, callback);
      }

      render() {
        return render({ ...this.props, ...this.renderVals() });
      }
    }

    // The bundled canvas defines Component and page.py appends the bridge subclass.
    const Component = new Function('DCLogic', `'use strict';\n${script.textContent}\nreturn Component;`)(DCLogic);
    const defaults = {};
    for (const [name, descriptor] of Object.entries(JSON.parse(script.dataset.props))) {
      if (Object.hasOwn(descriptor, 'default')) defaults[name] = descriptor.default;
    }

    for (const style of canvas.querySelectorAll('helmet style')) document.head.append(style);
    const pseudoStyles = document.createElement('style');
    pseudoStyles.textContent = rules.join('\n');
    document.head.append(pseudoStyles);

    canvas.replaceChildren();
    canvas.hidden = false;
    // A legacy root keeps every update in one synchronous lane, in call order: a click
    // rendered between a timer's update and its commit cannot roll this.state back.
    ReactDOM.render(React.createElement(Component, defaults), canvas);
    window.addEventListener('pagehide', function unmount(event) {
      if (event.persisted) return;
      ReactDOM.unmountComponentAtNode(canvas);
      window.removeEventListener('pagehide', unmount);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
  else mount();
})();

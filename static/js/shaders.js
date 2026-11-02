// GLSL sources for the point layers and trail composition passes.

export const TRAIL_VERT = `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `;

export const TRAIL_BLEND_FRAG = `
      varying vec2 vUv;
      uniform sampler2D uCurrent;
      uniform sampler2D uPrev;
      uniform float uPersistence;
      void main() {
        vec4 current = texture2D(uCurrent, vUv);
        vec4 prev = texture2D(uPrev, vUv) * uPersistence;
        gl_FragColor = max(current, prev);
      }
    `;

export const TRAIL_PRESENT_FRAG = `
      varying vec2 vUv;
      uniform sampler2D uTrail;
      void main() {
        gl_FragColor = texture2D(uTrail, vUv);
      }
    `;

export const VERT = `
      attribute float size;
      attribute float scanFlag;
      attribute float orderNorm;
      varying vec3 vColor;
      varying float vScanAlpha;
      varying float vScanBoost;
      uniform float uTime;
      uniform float uSizeMult;
      uniform float uReveal;
      void main() {
        vColor = color;
        vScanAlpha = 1.0;
        vScanBoost = 1.0;
        float scanSize = 1.0;
        float strongPulse = 1.0 + 0.26 * sin(uTime * 5.0 + position.x * 0.04 + position.z * 0.04);

        if (scanFlag > 0.5 && scanFlag < 1.5) {
          vScanAlpha = 0.14;
          vScanBoost = 0.25;
          scanSize = 0.8;
        } else if (scanFlag >= 1.5 && scanFlag < 2.5) {
          vScanAlpha = 1.0;
          vScanBoost = 1.55;
          scanSize = 1.35;
        } else if (scanFlag >= 2.5 && scanFlag < 3.5) {
          vScanAlpha = 1.0;
          vScanBoost = 2.0 * strongPulse;
          scanSize = 1.6 * strongPulse;
        } else if (scanFlag >= 3.5) {
          vScanAlpha = 1.0;
          vScanBoost = 2.8 * strongPulse;
          scanSize = 2.2 * strongPulse;
        }

        if (orderNorm > uReveal) {
          gl_PointSize = 0.0;
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          return;
        }

        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        if (scanFlag >= 3.5) {
          vec3 towardCamera = normalize(-mv.xyz);
          mv.xyz += towardCamera * 18.0;
        }
        float vDist = -mv.z;
        float pulse = 1.0 + 0.12 * sin(uTime * 1.8 + position.x * 0.07 + position.z * 0.05);
        gl_PointSize = size * scanSize * uSizeMult * pulse * (400.0 / max(vDist, 1.0));
        gl_PointSize = clamp(gl_PointSize, 1.0, 512.0);
        gl_Position  = projectionMatrix * mv;
      }
    `;

export const CORE_FRAG = `
      varying vec3 vColor;
      varying float vScanAlpha;
      varying float vScanBoost;
      uniform float uOpacity;
      uniform float uBloom;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;

        float coreMask = 1.0 - smoothstep(0.17, 0.19, d);
        float ringMask = smoothstep(0.18, 0.20, d) * (1.0 - smoothstep(0.45, 0.47, d));
        float rimMask = smoothstep(0.45, 0.47, d) * (1.0 - smoothstep(0.50, 0.51, d));

        vec3 coreCol = vec3(1.0) * uBloom;
        vec3 ringCol = mix(vColor, vec3(1.0), 0.65) * uBloom;
        vec3 rimCol = vColor * 0.5;

        vec3 col = (coreCol * coreMask + ringCol * ringMask + rimCol * rimMask) * vScanBoost;
        float alpha = (coreMask + ringMask + rimMask * 0.8) * uOpacity * vScanAlpha;
        gl_FragColor = vec4(col, alpha);
      }
    `;

export const HAZE_FRAG = `
      varying vec3 vColor;
      varying float vScanAlpha;
      varying float vScanBoost;
      uniform float uOpacity;
      uniform float uBloom;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float emit = pow(max(1.0 - d * 2.0, 0.0), 2.2);
        vec3 col = vColor * emit * uBloom * vScanBoost;
        gl_FragColor = vec4(col, emit * uOpacity * vScanAlpha);
      }
    `;

export const GLOW_FRAG = `
      varying vec3 vColor;
      varying float vScanAlpha;
      varying float vScanBoost;
      uniform float uOpacity;
      uniform float uBloom;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float g = exp(-d * d * 14.0);
        vec3 col = vColor * g * uBloom * vScanBoost;
        gl_FragColor = vec4(col, g * uOpacity * vScanAlpha);
      }
    `;

export const NEBULA_FRAG = `
      varying vec3 vColor;
      varying float vScanAlpha;
      varying float vScanBoost;
      uniform float uOpacity;
      uniform float uBloom;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float g = exp(-d * d * 4.0);
        vec3 col = vColor * g * uBloom * vScanBoost;
        gl_FragColor = vec4(col, g * uOpacity * vScanAlpha);
      }
    `;

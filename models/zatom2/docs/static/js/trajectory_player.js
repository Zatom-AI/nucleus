/**
 * Zatom-2 Interactive MolStar Trajectory Player
 * Handles multi-model mmCIF flow matching trajectories across molecules, materials, and proteins.
 */

(function () {
  'use strict';

  // Domain configurations
  const DOMAINS = {
    molecule: {
      id: 'molecule',
      label: 'Molecule (OMol25)',
      icon: 'fa-flask',
      formula: 'C₉H₂₀N₂ (31 atoms)',
      checkpoint: 'paper-table1-omol25-omat24-joint-properties-force-conditioned',
      checkpointUrl: 'https://huggingface.co/Zatom-AI/Zatom-2/tree/main/paper-ready-checkpoints/paper-table1-omol25-omat24-joint-properties-force-conditioned',
      description: 'Flow matching generation of a biomolecule sampled from the OMol25 distribution. Starting from Gaussian coordinate noise at t = 0, Zatom-2 transports mass along continuous velocity fields to form stable chemical valence at t = 1.',
      files: {
        denoised: 'static/trajectories/molecule_omol25_denoised.cif',
        noisy: 'static/trajectories/molecule_omol25_noisy.cif',
        final: 'static/trajectories/molecule_omol25_final.cif'
      },
      format: 'mmcif',
      totalFrames: 49,
      cameraRadius: {
        denoised: 8.5,
        noisy: 45.0,
        final: 6.5
      },
      representation: {
        type: 'ball-and-stick',
        params: { sizeFactor: 0.35, quality: 'highest' },
        color: 'element-symbol'
      },
      hasUnitCell: false,
      dynamicBonds: true
    },
    material: {
      id: 'material',
      label: 'Periodic Material (OMat24)',
      icon: 'fa-gem',
      formula: 'AlBe₃CoNi₄V (10 atoms / unit cell)',
      checkpoint: 'paper-table1-omol25-omat24-joint-properties-force-conditioned',
      checkpointUrl: 'https://huggingface.co/Zatom-AI/Zatom-2/tree/main/paper-ready-checkpoints/paper-table1-omol25-omat24-joint-properties-force-conditioned',
      description: 'Flow matching generation of a 3D periodic material lattice from the OMat24 distribution. Zatom-2 co-generates atomic Cartesian coordinates and fractional lattice parameters with unit cell volume tracking.',
      files: {
        denoised: 'static/trajectories/material_omat24_denoised.cif',
        noisy: 'static/trajectories/material_omat24_noisy.cif',
        final: 'static/trajectories/material_omat24_final.cif'
      },
      format: 'mmcif',
      totalFrames: 49,
      cameraRadius: {
        denoised: 8.0,
        noisy: 42.0,
        final: 6.0
      },
      representation: {
        type: 'spacefill',
        params: { sizeFactor: 0.38, quality: 'highest' },
        color: 'element-symbol'
      },
      hasUnitCell: true,
      dynamicBonds: false
    },
    protein: {
      id: 'protein',
      isPolymer: true,
      reversedModels: true,
      label: 'Protein Biomolecule',
      icon: 'fa-dna',
      formula: 'Monomer (256 residues)',
      checkpoint: 'paper-table3-scope-from-joint-properties-force-conditioned',
      checkpointUrl: 'https://huggingface.co/Zatom-AI/Zatom-2/tree/main/paper-ready-checkpoints/paper-table3-scope-from-joint-properties-force-conditioned',
      description: 'De novo all-atom protein generation transferred from molecular + material multitask generative-MLIP pretraining. Length-extrapolated backbone achieves 8/8 SolubleMPNN + ESMFold2 designability (scRMSD 0.73 Å, scTM 0.985) and structural novelty (train TM 0.35).',
      files: {
        denoised: 'static/trajectories/protein_scope_denoised.cif',
        noisy: 'static/trajectories/protein_scope_noisy.cif',
        final: 'static/trajectories/protein_scope_final.cif'
      },
      format: 'mmcif',
      totalFrames: 49,
      cameraRadius: {
        denoised: 48.0,
        noisy: 75.0,
        final: 48.0
      },
      representation: {
        type: 'cartoon',
        params: { quality: 'highest' },
        color: 'sequence-id',
        colorParams: {
          list: {
            kind: 'interpolate',
            colors: [0x3155a4, 0x26a6c7, 0x58b86b, 0xf2d34f, 0xd63b36]
          }
        }
      },
      hasUnitCell: false,
      dynamicBonds: true
    }
  };

  // State
  let viewer = null;
  let currentDomainKey = 'molecule';
  let currentMode = 'denoised'; // 'denoised' | 'noisy' | 'final'
  let currentFrameIndex = 0;
  let totalFrames = 49;
  let isPlaying = false;
  let playbackSpeed = 1.0;
  let playInterval = null;
  let currentModelRef = null;
  let currentTrajectory = null;
  let isSpinning = false;
  let isDwell = false;
  let dwellTimeout = null;
  let isUpdatingFrame = false;
  let nextFrameToUpdate = null;

  function clearDwell() {
    if (dwellTimeout) {
      clearTimeout(dwellTimeout);
      dwellTimeout = null;
    }
    isDwell = false;
  }

  // DOM Elements
  let dom = {};

  async function initPlayer() {
    // Cache DOM
    dom.container = document.getElementById('molstar-container') || document.querySelector('.molstar-container');
    dom.viewport = document.getElementById('molstar-app');
    dom.loading = document.getElementById('viewer-loading');
    dom.badgeTitle = document.getElementById('badge-title');
    dom.badgeStep = document.getElementById('badge-step');
    dom.badgeInfo = document.getElementById('badge-info');
    dom.playBtn = document.getElementById('btn-play') || document.getElementById('btn-start');
    dom.playIcon = document.getElementById('icon-play') || document.getElementById('icon-start');
    dom.stopBtn = document.getElementById('btn-stop');
    dom.prevBtn = document.getElementById('btn-prev');
    dom.nextBtn = document.getElementById('btn-next');
    dom.slider = document.getElementById('timeline-slider');
    dom.timeLabel = document.getElementById('timeline-label');
    dom.cardTitle = document.getElementById('card-domain-title');
    dom.cardDesc = document.getElementById('card-domain-desc');
    dom.cardCkpt = document.getElementById('card-domain-ckpt');
    dom.btnRefresh = document.getElementById('btn-refresh') || document.getElementById('btn-spin');
    dom.btnFullscreen = document.getElementById('btn-fullscreen') || document.getElementById('btn-reset-cam');

    // Setup domain tab listeners
    document.querySelectorAll('.domain-tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const domain = btn.getAttribute('data-domain');
        if (domain && domain !== currentDomainKey) {
          switchDomain(domain);
        }
      });
    });

    // Setup mode pills listeners
    document.querySelectorAll('.mode-pill-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const mode = btn.getAttribute('data-mode');
        if (mode && mode !== currentMode) {
          switchMode(mode);
        }
      });
    });

    // Setup speed pills listeners
    document.querySelectorAll('.speed-pill-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const speed = parseFloat(btn.getAttribute('data-speed'));
        if (speed) {
          setPlaybackSpeed(speed, btn);
        }
      });
    });

    // Setup player controls
    if (dom.playBtn) dom.playBtn.addEventListener('click', togglePlay);
    if (dom.stopBtn) dom.stopBtn.addEventListener('click', stop);
    if (dom.prevBtn) dom.prevBtn.addEventListener('click', () => stepFrame(-1));
    if (dom.nextBtn) dom.nextBtn.addEventListener('click', () => stepFrame(1));
    if (dom.slider) {
      dom.slider.addEventListener('input', (e) => {
        pause();
        const frame = parseInt(e.target.value, 10);
        goToFrame(frame);
      });
    }

    if (dom.btnRefresh) dom.btnRefresh.addEventListener('click', refreshTrajectory);
    if (dom.btnFullscreen) dom.btnFullscreen.addEventListener('click', toggleFullscreen);

    // Fullscreen change events
    ['fullscreenchange', 'webkitfullscreenchange', 'mozfullscreenchange', 'MSFullscreenChange'].forEach(evt => {
      document.addEventListener(evt, updateFullscreenUi);
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isFullscreenActive()) {
        toggleFullscreen();
      }
    });

    // Initialize MolStar Viewer
    try {
      showLoading(true);
      viewer = await molstar.Viewer.create('molstar-app', {
        layoutIsExpanded: false,
        layoutShowControls: false,
        layoutShowRemoteState: false,
        layoutShowSequence: false,
        layoutShowLog: false,
        viewportShowExpand: false,
        viewportShowControls: true,
        viewportShowSelectionMode: false,
        viewportShowAnimation: true, // Native Mol* trajectory bar
        viewportBackgroundColor: 'white'
      });

      viewer.plugin.canvas3d.setProps({
        renderer: { backgroundColor: 0xffffff },
        camera: { mode: 'orthographic', manualReset: true },
        cameraClipping: { radius: 100, far: false, minNear: 0.1 },
        cameraFog: { name: 'off', params: {} },
        sceneRadiusFactor: 3.0,
        helper: { axes: { name: 'off', params: {} } }
      });

      // Load initial domain
      await loadCurrentTrajectory();
    } catch (err) {
      console.error('Failed to initialize MolStar viewer:', err);
      if (dom.loading) {
        dom.loading.innerHTML = `<div class="notification is-danger is-light">Failed to initialize 3D viewer: ${err.message}</div>`;
      }
    }
  }

  function showLoading(show) {
    if (dom.loading) {
      if (show) dom.loading.classList.remove('is-hidden');
      else dom.loading.classList.add('is-hidden');
    }
  }

  function getModelIndexForFrame(frameIndex, domain, mode) {
    if (mode === 'final' || totalFrames <= 1) return 0;
    const isReversed = Boolean(domain && (domain.isPolymer || domain.reversedModels));
    return isReversed ? (totalFrames - 1 - frameIndex) : frameIndex;
  }

  async function loadCurrentTrajectory() {
    showLoading(true);
    pause();

    const domain = DOMAINS[currentDomainKey];
    totalFrames = (currentMode === 'final') ? 1 : domain.totalFrames;
    currentFrameIndex = 0;

    // Update UI labels
    updateMetadataDisplay();
    updateTimelineDisplay();

    if (dom.slider) {
      dom.slider.max = totalFrames - 1;
      dom.slider.value = 0;
      dom.slider.disabled = (currentMode === 'final');
    }
    if (dom.playBtn) dom.playBtn.disabled = (currentMode === 'final');
    if (dom.stopBtn) dom.stopBtn.disabled = (currentMode === 'final');
    if (dom.prevBtn) dom.prevBtn.disabled = (currentMode === 'final');
    if (dom.nextBtn) dom.nextBtn.disabled = (currentMode === 'final');

    const fileUrl = domain.files[currentMode] || domain.files.denoised;

    try {
      const p = viewer.plugin;

      // 1. Wipe all render objects from canvas3d (removes lingering unit cell wireframes)
      if (p.canvas3d) {
        p.canvas3d.clear();
      }

      // 2. Clear state tree and reset canvas properties
      await p.clear(true);

      // 3. Re-apply viewport canvas props (white background, orthographic camera)
      if (p.canvas3d) {
        p.canvas3d.setProps({
          renderer: { backgroundColor: 0xffffff },
          camera: { mode: 'orthographic', manualReset: true },
          cameraClipping: { radius: 100, far: false, minNear: 0.1 },
          cameraFog: { name: 'off', params: {} },
          sceneRadiusFactor: 3.0,
          helper: { axes: { name: 'off', params: {} } }
        });
      }

      const data = await p.builders.data.download({ url: fileUrl, isBinary: false });
      currentTrajectory = await p.builders.structure.parseTrajectory(data, domain.format);

      // Create model initially at the final folded structure so topology and representations initialize cleanly.
      // For polymer trajectories (reversedModels: true), Model 1 in the mmCIF (modelIndex 0)
      // is the clean folded structure, guaranteeing Mol* evaluates polymer continuity on valid
      // peptide bond coordinates and initializes an unbroken, 0-gap cartoon ribbon and secondary structures.
      // For standard non-reversed trajectories, modelIndex (totalFrames - 1) is the final state.
      const isReversed = Boolean(domain.isPolymer || domain.reversedModels);
      const targetInitModelIndex = isReversed ? 0 : ((totalFrames > 1) ? (totalFrames - 1) : 0);
      const model = await p.builders.structure.createModel(currentTrajectory, { modelIndex: targetInitModelIndex });
      currentModelRef = model.ref;

      // Create structure with Dynamic Bonds enabled by default for molecule and protein animations
      const isDynamicBonds = (domain.dynamicBonds !== undefined)
        ? domain.dynamicBonds
        : (domain.id === 'molecule' || domain.id === 'protein');

      const structure = await p.builders.structure.createStructure(
        model,
        domain.id === 'material'
          ? { name: 'symmetry', params: { ijkMin: [0, 0, 0], ijkMax: [1, 1, 1] } }
          : { name: 'model', params: { dynamicBonds: isDynamicBonds } }
      );

      // Create component
      const component = await p.builders.structure.tryCreateComponentStatic(
        structure,
        domain.id === 'protein' ? 'polymer' : 'all'
      );

      // Add main representation
      const reprProps = {
        type: domain.representation.type,
        typeParams: domain.representation.params || {},
        color: domain.representation.color,
        colorParams: domain.representation.colorParams || {}
      };
      await p.builders.structure.representation.addRepresentation(component, reprProps);

      // For protein: add C-alpha trace beads so coordinates remain fully visible throughout diffusion
      if (domain.id === 'protein') {
        try {
          const traceComp = await p.builders.structure.tryCreateComponentStatic(structure, 'trace');
          if (traceComp) {
            await p.builders.structure.representation.addRepresentation(traceComp, {
              type: 'spacefill',
              typeParams: { sizeFactor: 0.28, quality: 'highest' },
              color: 'sequence-id',
              colorParams: domain.representation.colorParams || {}
            });
          }
        } catch (e) {
          console.warn('Protein trace representation skipped:', e);
        }
      }

      // Add unit cell ONLY for materials
      if (domain.hasUnitCell) {
        try {
          await p.builders.structure.tryCreateUnitcell(model, {
            cellColor: 0x515967,
            cellScale: 1.2,
            ref: 'origin',
            attachment: 'corner'
          });
        } catch (e) {
          console.warn('Unit cell skipped:', e);
        }
      }

      // Frame camera: focus on origin with appropriate radius for current domain and mode
      const targetRadius = (domain.cameraRadius && domain.cameraRadius[currentMode])
        ? domain.cameraRadius[currentMode]
        : (domain.id === 'protein' ? 48.0 : 6.5);

      p.managers.camera.focusSphere(
        { center: [0, 0, 0], radius: targetRadius },
        { durationMs: 0 }
      );

      // Set model to frame 0 (t = 0.00, initial state) before playing forward
      const frame0ModelIndex = getModelIndexForFrame(0, domain, currentMode);
      if (targetInitModelIndex !== frame0ModelIndex) {
        await goToFrame(0);
      }

      showLoading(false);

      // Start auto-play forward for trajectories
      if (currentMode !== 'final') {
        start();
      } else {
        updatePlaybackUi();
      }
    } catch (err) {
      console.error('Error loading trajectory:', err);
      showLoading(false);
      if (dom.loading) {
        dom.loading.innerHTML = `<div class="notification is-warning is-light">Could not load ${domain.label} trajectory: ${err.message}</div>`;
      }
    }
  }

  async function goToFrame(frameIndex) {
    if (frameIndex < 0 || frameIndex >= totalFrames) return;
    currentFrameIndex = frameIndex;
    updateTimelineDisplay();
    updatePlaybackUi();

    if (!viewer || !currentModelRef) return;

    if (isUpdatingFrame) {
      nextFrameToUpdate = frameIndex;
      return;
    }

    isUpdatingFrame = true;
    try {
      const p = viewer.plugin;
      const domain = DOMAINS[currentDomainKey];
      const targetModelIndex = getModelIndexForFrame(currentFrameIndex, domain, currentMode);
      const update = p.state.data.build().to(currentModelRef).update({ modelIndex: targetModelIndex });
      await p.runTask(p.state.data.updateTree(update));

      // In the final frame (t = 1.00), ensure the camera focuses properly on the entire structure
      if (currentFrameIndex === totalFrames - 1) {
        const finalRadius = (domain && domain.cameraRadius)
          ? (domain.id === 'protein' ? 48.0 : (domain.cameraRadius.final || domain.cameraRadius[currentMode]))
          : 6.5;
        p.managers.camera.focusSphere(
          { center: [0, 0, 0], radius: finalRadius },
          { durationMs: 0 }
        );
      } else if (currentMode === 'noisy' && domain.cameraRadius && domain.cameraRadius.noisy) {
        p.managers.camera.focusSphere(
          { center: [0, 0, 0], radius: domain.cameraRadius.noisy },
          { durationMs: 0 }
        );
      }
    } catch (err) {
      console.warn('Frame update error:', err);
    } finally {
      isUpdatingFrame = false;
      if (nextFrameToUpdate !== null) {
        const target = nextFrameToUpdate;
        nextFrameToUpdate = null;
        if (target !== currentFrameIndex) {
          goToFrame(target);
        }
      }
    }
  }

  function stepFrame(delta) {
    pause();
    clearDwell();
    let next = (currentFrameIndex + delta) % totalFrames;
    if (next < 0) next += totalFrames;
    goToFrame(next);
  }

  function updatePlaybackUi() {
    if (dom.playBtn) {
      dom.playBtn.classList.toggle('is-active', isPlaying);
      dom.playBtn.title = isPlaying ? 'Pause Animation' : 'Play Animation';
      dom.playBtn.setAttribute('aria-label', isPlaying ? 'Pause Animation' : 'Play Animation');
    }
    if (dom.stopBtn) {
      dom.stopBtn.classList.remove('is-active');
    }
    if (dom.playIcon) {
      if (isPlaying) {
        dom.playIcon.classList.remove('fa-play');
        dom.playIcon.classList.add('fa-pause');
      } else {
        dom.playIcon.classList.remove('fa-pause');
        dom.playIcon.classList.add('fa-play');
      }
    }
  }

  function start() {
    if (currentMode === 'final') return;
    clearDwell();
    if (currentFrameIndex >= totalFrames - 1) {
      goToFrame(0);
    }
    play();
  }

  function play() {
    if (currentMode === 'final') return;
    isPlaying = true;
    updatePlaybackUi();

    clearInterval(playInterval);
    clearDwell();
    const intervalMs = Math.max(30, Math.round(75 / playbackSpeed));
    playInterval = setInterval(() => {
      if (isDwell) return;
      if (currentFrameIndex >= totalFrames - 1) {
        // Hold briefly on the final frame so viewers can inspect the predicted element types
        isDwell = true;
        const dwellMs = Math.round(1200 / playbackSpeed);
        dwellTimeout = setTimeout(() => {
          isDwell = false;
          if (isPlaying) {
            goToFrame(0);
          }
        }, dwellMs);
      } else {
        goToFrame(currentFrameIndex + 1);
      }
    }, intervalMs);
  }

  function pause() {
    isPlaying = false;
    updatePlaybackUi();
    clearInterval(playInterval);
    playInterval = null;
    clearDwell();
  }

  function stop() {
    pause();
    goToFrame(0);
    updatePlaybackUi();
  }

  function togglePlay() {
    if (isPlaying) pause();
    else start();
  }

  function setPlaybackSpeed(speed, activeBtn) {
    playbackSpeed = speed;
    document.querySelectorAll('.speed-pill-btn').forEach(btn => btn.classList.remove('is-active'));
    if (activeBtn) activeBtn.classList.add('is-active');
    if (isPlaying) {
      play(); // restarts interval with updated speed
    }
  }

  function switchDomain(domainKey) {
    if (!DOMAINS[domainKey]) return;
    currentDomainKey = domainKey;

    document.querySelectorAll('.domain-tab-btn').forEach(btn => {
      if (btn.getAttribute('data-domain') === domainKey) {
        btn.classList.add('is-active');
      } else {
        btn.classList.remove('is-active');
      }
    });

    loadCurrentTrajectory();
  }

  function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-pill-btn').forEach(btn => {
      if (btn.getAttribute('data-mode') === mode) {
        btn.classList.add('is-active');
      } else {
        btn.classList.remove('is-active');
      }
    });

    loadCurrentTrajectory();
  }

  async function refreshTrajectory() {
    if (dom.btnRefresh) dom.btnRefresh.classList.add('is-active');
    const icon = document.getElementById('icon-refresh');
    if (icon) icon.classList.add('fa-spin');
    try {
      if (isSpinning) {
        toggleSpin();
      }
      await loadCurrentTrajectory();
    } finally {
      if (icon) icon.classList.remove('fa-spin');
      if (dom.btnRefresh) dom.btnRefresh.classList.remove('is-active');
    }
  }

  function isFullscreenActive() {
    return !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement ||
      (dom.container && dom.container.classList.contains('is-fullscreen'))
    );
  }

  async function toggleFullscreen() {
    const container = dom.container || document.querySelector('.molstar-container');
    if (!container) return;

    if (!isFullscreenActive()) {
      try {
        if (container.requestFullscreen) {
          await container.requestFullscreen();
        } else if (container.webkitRequestFullscreen) {
          await container.webkitRequestFullscreen();
        } else if (container.msRequestFullscreen) {
          await container.msRequestFullscreen();
        } else {
          container.classList.add('is-fullscreen');
          document.documentElement.classList.add('is-clipped');
          updateFullscreenUi();
        }
      } catch (err) {
        console.warn('Native fullscreen failed, using CSS fallback:', err);
        container.classList.add('is-fullscreen');
        document.documentElement.classList.add('is-clipped');
        updateFullscreenUi();
      }
    } else {
      try {
        if (document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement) {
          if (document.exitFullscreen) {
            await document.exitFullscreen();
          } else if (document.webkitExitFullscreen) {
            await document.webkitExitFullscreen();
          } else if (document.msExitFullscreen) {
            await document.msExitFullscreen();
          }
        }
      } catch (err) {
        console.warn('Error exiting fullscreen:', err);
      }
      container.classList.remove('is-fullscreen');
      document.documentElement.classList.remove('is-clipped');
      updateFullscreenUi();
    }
  }

  function updateFullscreenUi() {
    const active = isFullscreenActive();
    if (dom.btnFullscreen) {
      dom.btnFullscreen.classList.toggle('is-active', active);
      dom.btnFullscreen.title = active ? 'Exit Fullscreen' : 'Toggle Fullscreen';
      dom.btnFullscreen.setAttribute('aria-label', active ? 'Exit Fullscreen' : 'Toggle Fullscreen');
    }
    const icon = document.getElementById('icon-fullscreen');
    if (icon) {
      if (active) {
        icon.classList.remove('fa-expand');
        icon.classList.add('fa-compress');
      } else {
        icon.classList.remove('fa-compress');
        icon.classList.add('fa-expand');
      }
    }
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
      if (viewer && viewer.plugin && viewer.plugin.canvas3d) {
        viewer.plugin.canvas3d.handleResize();
      }
      resetCamera();
    }, 100);
  }

  function toggleSpin() {
    if (!viewer) return;
    isSpinning = !isSpinning;
    viewer.plugin.canvas3d.setProps({
      trackball: { animate: isSpinning ? { name: 'spin', params: { speed: 1 } } : { name: 'off', params: {} } }
    });
    if (dom.btnRefresh) {
      dom.btnRefresh.classList.toggle('is-info', isSpinning);
    }
  }

  function resetCamera() {
    if (!viewer) return;
    const domain = DOMAINS[currentDomainKey];
    const targetRadius = (domain && domain.cameraRadius && domain.cameraRadius[currentMode])
      ? domain.cameraRadius[currentMode]
      : (domain.id === 'protein' ? 48.0 : 6.5);

    viewer.plugin.managers.camera.focusSphere(
      { center: [0, 0, 0], radius: targetRadius },
      { durationMs: 250 }
    );
  }

  function updateMetadataDisplay() {
    const domain = DOMAINS[currentDomainKey];
    if (dom.badgeTitle) {
      dom.badgeTitle.innerHTML = `<i class="fas ${domain.icon} mr-1"></i> ${domain.label}`;
    }
    if (dom.cardTitle) dom.cardTitle.textContent = `${domain.label}: ${domain.formula}`;
    if (dom.cardDesc) dom.cardDesc.textContent = domain.description;
    if (dom.cardCkpt) {
      dom.cardCkpt.innerHTML = `Checkpoint: <a href="${domain.checkpointUrl}" target="_blank" rel="noopener noreferrer" class="has-text-link">${domain.checkpoint}</a>`;
    }
  }

  function updateTimelineDisplay() {
    const domain = DOMAINS[currentDomainKey];
    const tVal = (totalFrames > 1) ? (currentFrameIndex / (totalFrames - 1)).toFixed(2) : '1.00';

    if (dom.slider) {
      dom.slider.value = currentFrameIndex;
    }
    if (dom.timeLabel) {
      dom.timeLabel.textContent = `t = ${tVal} (${currentFrameIndex + 1}/${totalFrames})`;
    }
    if (dom.badgeStep) {
      dom.badgeStep.textContent = `t = ${tVal} (Step ${currentFrameIndex + 1}/${totalFrames})`;
    }
    if (dom.badgeInfo) {
      const modeLabel = currentMode === 'denoised' ? 'Denoised (x̂₀|ₜ)' : (currentMode === 'noisy' ? 'Noisy (xₜ)' : 'Final');
      dom.badgeInfo.textContent = `${modeLabel} • ${domain.formula}`;
    }
  }

  // Initialize on load
  window.addEventListener('DOMContentLoaded', initPlayer);
})();

/**
 * Main Interactive Logic for Zatom-2 Project Page
 */

document.addEventListener('DOMContentLoaded', () => {
  // Results tabs switcher
  const resultsTabs = document.querySelectorAll('.results-tab-item');
  const resultsContents = document.querySelectorAll('.results-tab-content');

  resultsTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-target');

      resultsTabs.forEach(t => t.classList.remove('is-active'));
      resultsContents.forEach(c => c.classList.remove('is-active'));

      tab.classList.add('is-active');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('is-active');
      }
    });
  });

  // Representation sub-view switcher (Fig 16 vs Fig 15)
  const btnRepOmol = document.getElementById('btn-rep-omol25');
  const btnRepJoint = document.getElementById('btn-rep-joint');
  const viewRepOmol = document.getElementById('rep-view-omol25');
  const viewRepJoint = document.getElementById('rep-view-joint');

  if (btnRepOmol && btnRepJoint && viewRepOmol && viewRepJoint) {
    btnRepOmol.addEventListener('click', () => {
      btnRepOmol.classList.add('is-primary', 'is-selected');
      btnRepJoint.classList.remove('is-primary', 'is-selected');
      viewRepOmol.style.display = 'block';
      viewRepJoint.style.display = 'none';
    });

    btnRepJoint.addEventListener('click', () => {
      btnRepJoint.classList.add('is-primary', 'is-selected');
      btnRepOmol.classList.remove('is-primary', 'is-selected');
      viewRepJoint.style.display = 'block';
      viewRepOmol.style.display = 'none';
    });
  }

  // BibTeX Copy to clipboard
  const copyBtn = document.getElementById('btn-copy-bibtex');
  const bibtexCode = document.getElementById('bibtex-code');

  if (copyBtn && bibtexCode) {
    copyBtn.addEventListener('click', async () => {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(bibtexCode.innerText);
        } else {
          const textarea = document.createElement('textarea');
          textarea.value = bibtexCode.innerText;
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          document.body.removeChild(textarea);
        }
        const origText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check mr-1"></i> Copied!';
        copyBtn.classList.add('is-success');
        setTimeout(() => {
          copyBtn.innerHTML = origText;
          copyBtn.classList.remove('is-success');
        }, 2000);
      } catch (err) {
        console.error('Failed to copy text:', err);
      }
    });
  }

  // Initialize bulma-carousel if elements exist
  if (typeof bulmaCarousel !== 'undefined') {
    bulmaCarousel.attach('#gallery-carousel', {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      infinite: true,
      autoplay: false,
    });
  }

  // Gallery Image Zoom Modal (Lightbox)
  const zoomModal = document.getElementById('image-zoom-modal');
  const zoomImg = document.getElementById('zoom-modal-img');
  const zoomCaption = document.getElementById('zoom-modal-caption');
  const modalBgClose = document.getElementById('modal-bg-close');
  const modalBtnClose = document.getElementById('modal-btn-close');

  const openZoomModal = (src, caption) => {
    if (!zoomModal || !zoomImg) return;
    zoomImg.src = src;
    if (zoomCaption) {
      zoomCaption.innerHTML = caption || '';
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([zoomCaption]).catch(() => {});
      }
    }
    zoomModal.classList.add('is-active');
    document.documentElement.classList.add('is-clipped');
  };

  const closeZoomModal = () => {
    if (!zoomModal) return;
    zoomModal.classList.remove('is-active');
    document.documentElement.classList.remove('is-clipped');
    if (zoomImg) {
      zoomImg.src = '';
    }
  };

  document.querySelectorAll('.gallery-zoomable').forEach(item => {
    item.addEventListener('click', () => {
      const img = item.querySelector('img');
      const zoomSrc = item.getAttribute('data-zoom-src') || (img ? img.src : '');
      const caption = item.getAttribute('data-caption') || (img ? img.alt : '');
      if (zoomSrc) {
        openZoomModal(zoomSrc, caption);
      }
    });
  });

  if (modalBgClose) modalBgClose.addEventListener('click', closeZoomModal);
  if (modalBtnClose) modalBtnClose.addEventListener('click', closeZoomModal);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && zoomModal && zoomModal.classList.contains('is-active')) {
      closeZoomModal();
    }
  });
});


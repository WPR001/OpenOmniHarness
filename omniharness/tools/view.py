"""view: show up to 4 image files from the sandbox (contact sheets, frames, crops) within the image budget."""
import os

from .base import ToolInputError, register


@register('view',
          'Look at up to 4 image files from the sandbox (contact sheets, frames, crops).',
          {'type': 'object', 'properties': {
              'paths': {'type': 'array', 'items': {'type': 'string'}, 'description': 'image file names, at most 4'}},
           'required': ['paths']})
def view(box, args):
    from PIL import Image
    paths = args.get('paths') or args.get('path') or []
    if isinstance(paths, str):
        paths = [p.strip() for p in paths.replace(',', ' ').split() if p.strip()]
    paths = list(paths)[:4]
    if not paths:
        raise ToolInputError('give image file names in "paths"')
    left = box.cfg.max_images - box.images
    if left <= 0:
        raise ToolInputError(f'the image budget ({box.cfg.max_images}) is used up; answer with what you have')
    content, shown = [], []
    for name in paths[:left]:
        try:
            image = Image.open(box.sb.path(name)).convert('RGB')
        except (OSError, ValueError) as e:
            content.append({'type': 'text', 'text': f'{name}: cannot open ({e}). {nearby_images(box.sb, name)}'})
            continue
        if max(image.size) > 1920:
            image.thumbnail((1920, 1920), Image.BICUBIC)
        content += [{'type': 'text', 'text': f'{name}:'}, {'type': 'image', 'image': image}]
        shown.append(name)
    box.images += len(shown)
    note = f'(showing {len(shown)} image(s); {box.cfg.max_images - box.images} left in the budget)'
    return content + [{'type': 'text', 'text': note}]


def nearby_images(sandbox, name):
    """List the images that do exist next to a missing file, so the model can pick the right name."""
    folder = os.path.dirname(name) or '.'
    try:
        found = sorted(f for f in os.listdir(sandbox.path(folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    except (OSError, ValueError):
        return ''
    if not found:
        return f'No images in {folder}/.'
    shown = ', '.join(os.path.join(folder, f) if folder != '.' else f for f in found[:24])
    return f'Images in {folder}/: {shown}' + (' ...' if len(found) > 24 else '')

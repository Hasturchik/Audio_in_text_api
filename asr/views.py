import asyncio

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .audio_analis import process_audio

@csrf_exempt
async def asr_view(request):
    """
    обработчик запроса
    """
    if request.method == 'POST':
        audio_file = request.FILES.get('file')
        if not audio_file:
            return JsonResponse({"error": "No audio file provided"}, status=400)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_audio, audio_file)

        return JsonResponse(result, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 4})

    return JsonResponse({"error": "Invalid request method"}, status=405)
